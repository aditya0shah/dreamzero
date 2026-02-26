"""Websocket server for DreamZero offline evaluation.

Based on socket_test_optimized_AR.py with two key changes:
  1. Saves decoded video as separate per-camera-view files.
  2. Decodes and saves video immediately after every inference query.

Launch with torchrun:
  torchrun --nproc_per_node=N eval_server.py --model-path <path> --port 8000
"""

import dataclasses
import logging
import socket
import asyncio
import os
import http
import time
import traceback
import datetime
import pickle

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
import tyro
import numpy as np
import imageio
from einops import rearrange
from tianshou.data import Batch
import websockets.asyncio.server as _server
import websockets.frames

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000
    model_path: str = "/mnt/aws-lfs-01/shared/seonghyeony/checkpoints/dreamvla/1105/wan_action_train_i2v_multiview_agibot_diverse_subtask_subsampling_action_OTJ_1104_steps100000_gpus128_bs128_per_device1_shared_time_multiview/copy-ckpt-26000"
    enable_dit_cache: bool = False
    max_chunk_size: int | None = None
    embodiment_tag: str = "xdof"
    # Per-view video splitting
    view_names: tuple[str, ...] = ("top_head", "hand_left", "hand_right")


# ---------------------------------------------------------------------------
# Video decode & per-view splitting helpers
# ---------------------------------------------------------------------------
def decode_latents_to_frames(policy: GrootSimPolicy, latents: torch.Tensor) -> np.ndarray:
    """Decode latent video tensor to uint8 pixel frames.

    Args:
        policy: The loaded GrootSimPolicy.
        latents: (B, C, T, H, W) latent tensor.

    Returns:
        (T_out, H_out, W_out, 3) uint8 numpy array.
    """
    ah = policy.trained_model.action_head
    frames = ah.vae.decode(
        latents,
        tiled=ah.tiled,
        tile_size=(ah.tile_size_height, ah.tile_size_width),
        tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
    )
    # (B, C, T, H, W) -> (B, T, H, W, C)
    frames = rearrange(frames, "B C T H W -> B T H W C")
    frames = ((frames[0].float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return frames


def split_mosaic_into_views(
    frames: np.ndarray,
    view_names: tuple[str, ...] | list[str],
) -> dict[str, np.ndarray]:
    """Split a 2×2 mosaic back into individual camera views.

    The mosaic layout (from DreamTransform._prepare_video):
        +------------+-------------+
        | view 0     | view 2      |
        | (top_head) | (hand_right)|
        +------------+-------------+
        | view 1     | BLACK       |
        | (hand_left)| (zeros)     |
        +------------+-------------+

    Args:
        frames: (T, H_mosaic, W_mosaic, C) uint8 array.
        view_names: ordered list, e.g. ("top_head", "hand_left", "hand_right").

    Returns:
        Dict mapping view name -> (T, h, w, C) uint8 array.
    """
    _, H, W, _ = frames.shape
    h, w = H // 2, W // 2

    views: dict[str, np.ndarray] = {}
    if len(view_names) >= 1:
        views[view_names[0]] = frames[:, :h, :w, :]       # top-left
    if len(view_names) >= 2:
        views[view_names[1]] = frames[:, h:, :w, :]       # bottom-left
    if len(view_names) >= 3:
        views[view_names[2]] = frames[:, :h, w:, :]       # top-right
    return views


# ---------------------------------------------------------------------------
# Websocket policy server
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    """Serves a policy over websocket with distributed inference support."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
        view_names: tuple[str, ...] = ("top_head", "hand_left", "hand_right"),
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        self._signal_group = signal_group
        self._view_names = view_names

        self.video_across_time: list[torch.Tensor] = []
        self._msg_index = 0
        self._current_traj_id = ""
        self._segment_index = 0

        logging.getLogger("websockets.server").setLevel(logging.INFO)

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    # -- video save helpers -------------------------------------------------

    def _save_videos(self, latents_cat: torch.Tensor, video_name: str) -> None:
        """Decode latents and save per-view + mosaic videos.

        Args:
            latents_cat: (B, C, T, H, W) latent tensor.
            video_name: e.g. "traj_0/seg_000" — used as the file path under each folder.
        """
        if not self._output_dir:
            return

        frames_mosaic = decode_latents_to_frames(self._policy, latents_cat)
        save_dir = self._output_dir

        # Save mosaic video
        mosaic_path = os.path.join(save_dir, "mosaic", f"{video_name}.mp4")
        os.makedirs(os.path.dirname(mosaic_path), exist_ok=True)
        imageio.mimsave(mosaic_path, list(frames_mosaic), fps=5, codec="libx264")
        logger.info("Saved mosaic video to %s", mosaic_path)

        # Save per-view videos into per_view/<view_name>/
        views = split_mosaic_into_views(frames_mosaic, self._view_names)
        for vname, vframes in views.items():
            vpath = os.path.join(save_dir, "per_view", vname, f"{video_name}.mp4")
            os.makedirs(os.path.dirname(vpath), exist_ok=True)
            imageio.mimsave(vpath, list(vframes), fps=5, codec="libx264")
            logger.info("Saved %s video to %s", vname, vpath)

    # -- serve / run --------------------------------------------------------

    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    async def run(self, rank: int = 0):
        if rank == 0:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            await self._worker_loop()

    # -- distributed helpers ------------------------------------------------

    async def _worker_loop(self):
        """Non-rank-0 processes wait for signals and participate in inference."""
        logger.info("Worker loop started for rank %d", dist.get_rank())
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")

        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)
                signal = signal_tensor.item()

                if signal == 1:
                    logger.info("Rank %d received shutdown signal", dist.get_rank())
                    break
                elif signal == 2:
                    # Idle — client disconnected, wait for next one
                    continue

                # signal == 0: inference round
                batch = self._receive_batch_from_rank0()
                dist.barrier()
                with torch.no_grad():
                    self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception:
                logger.error("Worker loop error on rank %d:\n%s", dist.get_rank(), traceback.format_exc())
                break

    def _receive_batch_from_rank0(self) -> Batch:
        size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(data_tensor, src=0)
        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        serialized = pickle.dumps(obs)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    # -- websocket handler --------------------------------------------------

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
        ah = self._policy.trained_model.action_head

        try:
            while True:
                try:
                    # 1. Receive observation
                    recv_t = time.perf_counter()
                    data = await websocket.recv()
                    infer_t = time.perf_counter()
                    print(f"Wait Time: {infer_t - recv_t:.3f}s")

                    obs = msgpack_numpy.unpackb(data)
                    self._msg_index += 1

                    # Extract trajectory ID metadata (not passed to the model)
                    traj_id = obs.pop("__traj_id__", f"unknown")
                    if traj_id != self._current_traj_id:
                        # Flush remaining latents from previous trajectory
                        if self.video_across_time and self._current_traj_id:
                            remaining = torch.cat(self.video_across_time, dim=2)
                            seg_name = f"traj_{self._current_traj_id}/seg_{self._segment_index:03d}"
                            self._save_videos(remaining, seg_name)
                        self.video_across_time = []
                        self._current_traj_id = traj_id
                        self._segment_index = 0

                    # 2. Signal workers to participate (0 = continue)
                    signal_tensor.zero_()
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group)
                    self._broadcast_batch_to_workers(obs)

                    # 3. Distributed forward pass
                    batch = Batch(obs=obs)
                    dist.barrier()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()
                    print(f"Inference Time: {time.perf_counter() - infer_t:.3f}s")

                    # 4. Accumulate video latents
                    self.video_across_time.append(video_pred)

                    # 5. Check for autoregressive segment boundary
                    segment_boundary = (
                        ah.current_start_frame == 1 + ah.num_frame_per_block
                        and len(self.video_across_time) > 1
                    )

                    seg_name = f"traj_{self._current_traj_id}/seg_{self._segment_index:03d}"

                    if segment_boundary:
                        # Decode completed segment (all but latest chunk), save, advance
                        completed = torch.cat(self.video_across_time[:-1], dim=2)
                        self._save_videos(completed, seg_name)
                        self._segment_index += 1
                        self.video_across_time = [video_pred]
                    else:
                        # Decode everything so far — overwrite the same segment file
                        all_latents = torch.cat(self.video_across_time, dim=2)
                        self._save_videos(all_latents, seg_name)

                    # 6. Post-process actions (waist concat for agibot)
                    action_chunk = result_batch.act
                    if "action.waist_pitch" in dir(action_chunk):
                        waist_pitch = getattr(action_chunk, "action.waist_pitch", None)
                        waist_lift = getattr(action_chunk, "action.waist_lift", None)
                        if waist_pitch is not None and waist_lift is not None:
                            action_chunk["action.waist_position"] = np.concatenate(
                                (waist_pitch, waist_lift), axis=-1
                            )

                    # Convert Batch to plain dict
                    action_dict = {}
                    for k in dir(action_chunk):
                        if k.startswith("action."):
                            action_dict[k] = getattr(action_chunk, k)

                    # 7. Include segment metadata so client can align GT saving
                    action_dict["__segment_index__"] = self._segment_index
                    action_dict["__is_segment_boundary__"] = segment_boundary

                    # 8. Send actions back to client
                    await websocket.send(packer.pack(action_dict))

                except websockets.ConnectionClosed:
                    logger.info("Connection from %s closed", websocket.remote_address)
                    # Flush remaining latents as final segment
                    if self.video_across_time:
                        remaining = torch.cat(self.video_across_time, dim=2)
                        seg_name = f"traj_{self._current_traj_id}/seg_{self._segment_index:03d}"
                        self._save_videos(remaining, seg_name)
                    self.video_across_time = []
                    break

                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback in previous frame.",
                    )
                    raise

        finally:
            # Tell workers to go idle
            logger.info("Rank 0: session ended — sending idle signal (2) to workers")
            signal_tensor.fill_(2)
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)


# ---------------------------------------------------------------------------
# Distributed init & health check
# ---------------------------------------------------------------------------
def init_mesh() -> DeviceMesh:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))
    print(f"Rank {rank}/{world_size} (PID {os.getpid()}) on cuda:{rank}")
    return mesh


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 100

    policy_metadata = {
        "embodiment": args.embodiment_tag,
        "model_name": "dreamvla",
        "model_path": args.model_path,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info("Rank %d initialized gloo signal group", rank)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    # Build output directory
    if rank == 0:
        parent_dir = os.path.dirname(args.model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        ckpt_name = os.path.basename(args.model_path)
        output_dir = os.path.join(parent_dir, f"eval_gen_{date_suffix}", ckpt_name)
        logging.info("Videos will be saved to %s", output_dir)
    else:
        output_dir = None
        logging.info("Rank %d starting as worker", rank)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        output_dir=output_dir,
        signal_group=signal_group,
        view_names=args.view_names,
    )
    server.serve_forever(rank=rank)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
