#!/usr/bin/env python3
"""Websocket client for DreamZero offline evaluation.

Loads episodes from a LeRobot dataset and queries the eval_server at every
action-horizon interval, using the model's video subsampling pattern.

Supports two server types:
  - groot (default): step-by-step observation/action exchange with gr00t eval_server
  - dreamdojo: one-shot trajectory request to DreamDojo eval_server

Usage:
  # gr00t mode (default)
  python eval_client.py \
      --host <server_host> --port 8000 \
      --dataset-path /path/to/lerobot/dataset \
      --model-path /path/to/checkpoint \
      --embodiment-tag agibot \
      --num-trajectories 3

  # DreamDojo mode
  python eval_client.py \
      --server-type dreamdojo \
      --host <server_host> --port 8000 \
      --dataset-path /path/to/lerobot/dataset \
      --model-path /path/to/checkpoint \
      --embodiment-tag gr1 \
      --no-save-gt-video \
      --num-trajectories 3
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import imageio
import websockets
from openpi_client import msgpack_numpy
from omegaconf import OmegaConf
from hydra.utils import instantiate

from groot.vla.data.dataset.lerobot import LeRobotSingleDataset, ModalityConfig
from groot.vla.data.schema import EmbodimentTag

logger = logging.getLogger(__name__)

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600

# Video delta indices per embodiment.
# First inference: 1 frame at current step. Subsequent: 4 frames subsampled.
VIDEO_DELTA_INDICES = {
    "agibot": {"first": [0], "subsequent": [-47, -32, -16, 0]},
    "xdof": {"first": [0], "subsequent": [-47, -32, -16, 0]},
    "oxe_droid": {"first": [0], "subsequent": [-23, -16, -8, 0]},
}


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def load_train_cfg(model_path: str):
    """Load the training config from a checkpoint's experiment_cfg/conf.yaml."""
    conf_path = os.path.join(model_path, "experiment_cfg", "conf.yaml")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Config not found: {conf_path}")
    return OmegaConf.load(conf_path)


def load_modality_configs_from_checkpoint(train_cfg, embodiment_tag: str):
    """Extract modality_configs for the given embodiment from the training config."""
    if embodiment_tag in train_cfg.modality_configs:
        return instantiate(train_cfg.modality_configs[embodiment_tag])
    return instantiate(train_cfg.modality_configs)


def infer_metadata_version(train_cfg, embodiment_tag: str) -> str | None:
    """Read metadata_versions from the training config and return the version for this embodiment."""
    versions = OmegaConf.to_container(train_cfg.get("metadata_versions", {}), resolve=True)
    return versions.get(embodiment_tag)


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------

def build_obs_for_step(
    dataset: LeRobotSingleDataset,
    traj_id,
    step_index: int,
    inference_count: int,
    embodiment_tag: str,
) -> dict:
    """Build an observation dict for one inference step.

    Video delta indices follow the per-embodiment pattern:
      - inference 0  → [0]  (1 frame)
      - inference 1+ → e.g. [-47, -32, -16, 0] (4 frames)

    State / language use standard delta [0].
    Action keys are stripped (only observations are sent to the server).
    """
    delta_cfg = VIDEO_DELTA_INDICES.get(embodiment_tag, VIDEO_DELTA_INDICES["agibot"])
    video_deltas = delta_cfg["first"] if inference_count == 0 else delta_cfg["subsequent"]

    indices: dict[str, list[int] | np.ndarray] = {}
    for key, base_deltas in dataset.delta_indices.items():
        if key.startswith("action."):
            continue  # skip actions
        if key.startswith("video."):
            indices[key] = np.array(video_deltas, dtype=int) + step_index
        else:
            indices[key] = np.array([0], dtype=int) + step_index

    data = dataset.get_step_data(traj_id, indices)

    # Return only observation keys (no actions)
    return {k: v for k, v in data.items() if not k.startswith("action.")}


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

def compute_gt_frame_offsets(
    action_horizon: int, num_frames: int, num_frame_per_block: int, call_in_segment: int,
) -> list[int]:
    """Compute GT frame offsets for one call within an AR segment.

    One video segment has `num_latent = (num_frames-1)//4 + 1` latent frames.
    The first call decodes `1 + num_frame_per_block` latents → `1 + num_frame_per_block*4` pixels.
    Subsequent calls decode `num_frame_per_block` latents → `num_frame_per_block*4` pixels.
    Pixel frames are spaced `frame_step = action_horizon / (num_latent - 1)` real steps apart.

    Args:
        action_horizon: Action steps per inference call (e.g. 48).
        num_frames: Total pixel frames per segment (e.g. 33).
        num_frame_per_block: Latent frames generated per AR step (e.g. 2).
        call_in_segment: 0 for first call in segment, 1/2/3 for subsequent calls.

    Returns:
        List of integer offsets relative to the current step.
    """
    num_latent = (num_frames - 1) // 4 + 1          # e.g. 9
    frame_step = action_horizon // (num_latent - 1)  # e.g. 6

    if call_in_segment == 0:
        # First call: conditioning frame + num_frame_per_block latents
        # = 1 + num_frame_per_block*4 pixel frames (e.g. 9)
        n_pixels = 1 + num_frame_per_block * 4
        return [i * frame_step for i in range(n_pixels)]
    else:
        # Subsequent calls: num_frame_per_block latents = num_frame_per_block*4 pixel frames (e.g. 8)
        n_pixels = num_frame_per_block * 4
        return [(i + 1) * frame_step for i in range(n_pixels)]


def load_gt_video_frames(
    dataset: LeRobotSingleDataset,
    traj_id,
    step: int,
    traj_length: int,
    gt_offsets: list[int],
) -> dict[str, np.ndarray]:
    """Load GT video frames at the given offsets from the dataset.

    Returns:
        Dict mapping video key -> (T_gt, H, W, C) uint8 array.
    """
    abs_indices = np.array(gt_offsets, dtype=int) + step
    abs_indices = np.clip(abs_indices, 0, traj_length - 1)

    indices = {}
    for key in dataset.delta_indices:
        if key.startswith("video."):
            indices[key] = abs_indices

    data = dataset.get_step_data(traj_id, indices)
    return {k: v for k, v in data.items() if k.startswith("video.")}


async def run_trajectory(
    websocket,
    packer: msgpack_numpy.Packer,
    dataset: LeRobotSingleDataset,
    traj_id,
    traj_index: int,
    action_horizon: int,
    embodiment_tag: str,
    output_dir: str | None,
    num_frames: int,
    num_frame_per_block: int,
):
    """Send observations and collect actions for a single trajectory."""
    traj_length = int(dataset.trajectory_lengths[traj_index])
    logger.info("Trajectory %s: %d steps, action_horizon=%d", traj_id, traj_length, action_horizon)

    all_actions: list[dict] = []
    # GT frames accumulated for the current segment: {view_key: [array, ...]}
    gt_segment_acc: dict[str, list[np.ndarray]] = {}
    # Completed segments: {view_key: [(seg_idx, frames_array), ...]}
    gt_completed: dict[str, list[tuple[int, np.ndarray]]] = {}
    inference_count = 0
    # Track current server segment — updated from server response
    prev_server_seg = -1
    call_in_segment = 0

    # Start from a step where all video delta indices are valid.
    delta_cfg = VIDEO_DELTA_INDICES.get(embodiment_tag, VIDEO_DELTA_INDICES["agibot"])
    min_start = abs(min(delta_cfg["subsequent"]))
    start_step = min_start + 1

    for step in range(start_step, traj_length, action_horizon):
        obs = build_obs_for_step(dataset, traj_id, step, inference_count, embodiment_tag)

        # Log shapes on first send
        if inference_count == 0:
            for k, v in obs.items():
                shape = v.shape if isinstance(v, np.ndarray) else type(v).__name__
                logger.info("  %s: %s", k, shape)

        # Attach trajectory ID for the server to organize output files
        obs["__traj_id__"] = str(traj_id)

        t0 = time.time()
        await websocket.send(packer.pack(obs))
        action_raw = await websocket.recv()
        elapsed = time.time() - t0

        if isinstance(action_raw, str):
            raise RuntimeError(f"Server error:\n{action_raw}")

        action = msgpack_numpy.unpackb(action_raw)

        # Extract segment metadata from server response
        server_seg = int(action.pop("__segment_index__", 0))
        is_boundary = bool(action.pop("__is_segment_boundary__", False))

        # Detect segment transitions from the server's perspective
        if is_boundary or (server_seg != prev_server_seg and prev_server_seg >= 0):
            # The server just finalized a segment — flush our accumulated GT
            if gt_segment_acc:
                for k, chunks in gt_segment_acc.items():
                    concatenated = np.concatenate(chunks, axis=0)
                    if k not in gt_completed:
                        gt_completed[k] = []
                    gt_completed[k].append((prev_server_seg, concatenated))
                gt_segment_acc = {}
            call_in_segment = 0

        prev_server_seg = server_seg

        # Load GT frames for this call (first call in segment gets more frames)
        gt_offsets = compute_gt_frame_offsets(
            action_horizon, num_frames, num_frame_per_block, call_in_segment,
        )
        gt_frames = load_gt_video_frames(dataset, traj_id, step, traj_length, gt_offsets)
        for k, v in gt_frames.items():
            if k not in gt_segment_acc:
                gt_segment_acc[k] = []
            gt_segment_acc[k].append(v)

        logger.info(
            "  step %d  inference %d  server_seg %d  call_in_seg %d  boundary=%s  %.2fs",
            step, inference_count, server_seg, call_in_segment, is_boundary, elapsed,
        )

        call_in_segment += 1

        # Store per-step actions
        for j in range(min(action_horizon, traj_length - step)):
            step_action = {}
            for k, v in action.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1:
                    step_action[k] = v[j]
                else:
                    step_action[k] = v
            all_actions.append(step_action)

        inference_count += 1

    # Flush final segment
    if gt_segment_acc:
        for k, chunks in gt_segment_acc.items():
            concatenated = np.concatenate(chunks, axis=0)
            if k not in gt_completed:
                gt_completed[k] = []
            gt_completed[k].append((prev_server_seg, concatenated))

    # Save actions to disk
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"actions_traj_{traj_id}.npz")
        flat = {}
        for i, a in enumerate(all_actions):
            for k, v in a.items():
                flat[f"{i:06d}_{k}"] = v
        np.savez(save_path, **flat)
        logger.info("Saved %d action steps to %s", len(all_actions), save_path)

    # Save GT per-view videos — using server's segment indices for alignment
    if output_dir and gt_completed:
        gt_dir = os.path.join(output_dir, "gt_video")
        for view_key, segments in gt_completed.items():
            view_name = view_key.replace("video.", "")
            for seg_idx, seg_frames in segments:
                seg_dir = os.path.join(gt_dir, view_name, f"traj_{traj_id}")
                os.makedirs(seg_dir, exist_ok=True)
                vid_path = os.path.join(seg_dir, f"seg_{seg_idx:03d}.mp4")
                imageio.mimsave(vid_path, list(seg_frames), fps=5, codec="libx264")
            logger.info(
                "Saved GT %s: %d segments to %s",
                view_name, len(segments),
                os.path.join(gt_dir, view_name, f"traj_{traj_id}"),
            )

    return all_actions


# ---------------------------------------------------------------------------
# DreamDojo one-shot trajectory
# ---------------------------------------------------------------------------

async def run_trajectory_dreamdojo(
    websocket,
    packer: msgpack_numpy.Packer,
    dataset: LeRobotSingleDataset,
    traj_id,
    traj_index: int,
    action_horizon: int,
    embodiment_tag: str,
    num_frames: int,
    num_frame_per_block: int,
):
    """Send per-segment action chunks to DreamDojo server for video generation.

    To align with DreamZero segments, we iterate through the trajectory in
    segment-sized chunks.  Each DreamZero segment covers
    ``action_horizon * calls_per_segment`` raw action steps and produces
    ``num_frames`` video frames.

    For each segment we send the GT conditioning frame (at the segment start)
    plus the raw per-joint action arrays for that chunk.  The server handles
    normalization, 384-dim conversion, and video generation/saving.
    """
    traj_length = int(dataset.trajectory_lengths[traj_index])

    # Compute segment parameters matching DreamZero
    num_latent = (num_frames - 1) // 4 + 1  # e.g. 9
    calls_per_segment = (num_latent - 1) // num_frame_per_block  # e.g. 4
    segment_action_steps = action_horizon * calls_per_segment  # e.g. 48*4 = 192

    logger.info(
        "Trajectory %s: %d steps, segment=%d raw steps (%d calls × %d horizon)",
        traj_id, traj_length, segment_action_steps, calls_per_segment, action_horizon,
    )

    # Determine video keys for frame extraction
    video_keys = [k for k in dataset.delta_indices if k.startswith("video.")]
    action_keys = [k for k in dataset.delta_indices if k.startswith("action.")]

    segment_index = 0
    for seg_start in range(0, traj_length, segment_action_steps):
        seg_end = min(seg_start + segment_action_steps, traj_length)
        if seg_end - seg_start < 48: # TODO: fix hardcoding
            break  # need >= 12 raw steps for [::4] subsample + delta

        # --- Extract GT conditioning frame at segment start ---
        frame_data = dataset.get_step_data(
            traj_id, {k: np.array([seg_start]) for k in video_keys},
        )
        first_frame = None
        for k, v in frame_data.items():
            if k.startswith("video."):
                first_frame = v[0] if v.ndim == 4 else v  # [H, W, C]
                break
        if first_frame is None:
            raise ValueError(f"No video frame at step {seg_start} for traj {traj_id}")

        # --- Extract raw actions for this segment ---
        seg_indices = np.arange(seg_start, seg_end)
        action_data = dataset.get_step_data(
            traj_id, {k: seg_indices for k in action_keys},
        )
        raw_actions = {k: v for k, v in action_data.items() if k.startswith("action.")}

        # --- Send segment to server ---
        msg = {
            "traj_id": str(traj_id),
            "segment_index": segment_index,
            "first_frame": first_frame,
            "raw_actions": raw_actions,
            "embodiment_tag": embodiment_tag,
            "num_frames": num_frames,
        }

        t0 = time.time()
        await websocket.send(packer.pack(msg))

        ack_raw = await websocket.recv()
        elapsed = time.time() - t0

        if isinstance(ack_raw, str):
            raise RuntimeError(f"DreamDojo server error:\n{ack_raw}")

        ack = msgpack_numpy.unpackb(ack_raw)
        logger.info(
            "  seg %d  steps [%d:%d]  %d frames  %.2fs",
            segment_index, seg_start, seg_end,
            ack.get("num_frames", 0), elapsed,
        )

        segment_index += 1

    logger.info("  Trajectory %s: %d segments completed", traj_id, segment_index)


# ---------------------------------------------------------------------------
# Main client loop
# ---------------------------------------------------------------------------

async def eval_client(
    host: str,
    port: int,
    dataset_path: str,
    model_path: str,
    embodiment_tag: str,
    video_backend: str,
    action_horizon: int,
    num_trajectories: int,
    traj_indices: list[int] | None,
    output_dir: str | None,
    language_key: str | None,
    server_type: str = "groot",
    save_gt_video: bool = True,
):
    # 1. Load training config and modality configs from checkpoint
    logger.info("Loading config from %s", model_path)
    train_cfg = load_train_cfg(model_path)
    modality_configs = load_modality_configs_from_checkpoint(train_cfg, embodiment_tag)

    # Override language key if specified
    if language_key and hasattr(modality_configs, "language"):
        modality_configs.language.modality_keys = [language_key]

    # For DreamDojo mode, ensure the action modality includes all keys that
    # DreamDojo (groot_dreams) expects — the gr00t checkpoint config may not
    # include them since gr00t predicts actions rather than consuming them.
    if server_type == "dreamdojo":
        DREAMDOJO_ACTION_KEYS = {
            "xdof": [
                "ee_pose_action_left", "ee_pose_action_right",
                "gripper_pos_action_left", "gripper_pos_action_right",
                "joint_pos_action_left", "joint_pos_action_right",
            ],
            "gr1": [
                "left_arm", "right_arm", "left_hand", "right_hand", "waist",
            ],
            "g1": [
                "left_leg", "right_leg", "waist",
                "left_arm", "left_hand", "right_arm", "right_hand",
            ],
            "agibot": [
                "left_arm_joint_position", "right_arm_joint_position",
                "left_effector_position", "right_effector_position",
                "head_position", "waist_position", "robot_velocity",
            ],
        }
        if embodiment_tag in DREAMDOJO_ACTION_KEYS:
            action_keys = [f"action.{k}" for k in DREAMDOJO_ACTION_KEYS[embodiment_tag]]
            if hasattr(modality_configs, "action"):
                modality_configs.action.modality_keys = action_keys
            elif isinstance(modality_configs, dict) and "action" in modality_configs:
                modality_configs["action"].modality_keys = action_keys
            else:
                modality_configs["action"] = ModalityConfig(
                    delta_indices=[0], modality_keys=action_keys,
                )
            logger.info("DreamDojo action keys set: %s", action_keys)

    logger.info("Modality configs:\n%s", modality_configs)

    # Infer use_global_metadata and metadata_version from checkpoint config
    use_global_metadata = train_cfg.get("use_global_metadata", True)
    metadata_version = infer_metadata_version(train_cfg, embodiment_tag)
    if use_global_metadata and metadata_version is None:
        raise ValueError(
            f"use_global_metadata=True but could not infer metadata_version for '{embodiment_tag}'. "
            "Check metadata_versions in experiment_cfg/conf.yaml."
        )
    logger.info("use_global_metadata=%s, metadata_version='%s'", use_global_metadata, metadata_version)

    # Read video generation params from training config
    num_frames = int(train_cfg.get("num_frames", 33))
    num_frame_per_block = int(train_cfg.get("num_frame_per_block", 2))
    logger.info("num_frames=%d, num_frame_per_block=%d, action_horizon=%d",
                num_frames, num_frame_per_block, action_horizon)

    # 2. Load dataset
    logger.info("Loading dataset from %s", dataset_path)
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=EmbodimentTag(embodiment_tag),
        video_backend=video_backend,
        metadata_version=metadata_version,
        use_global_metadata=use_global_metadata,
        transforms=None,  # server handles normalization
    )

    n_trajs = len(dataset.trajectory_ids)
    logger.info("Dataset has %d trajectories, lengths=%s", n_trajs, dataset.trajectory_lengths)

    # Default output_dir to the same checkpoint directory the server uses
    if output_dir is None:
        import datetime
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        ckpt_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"eval_gen_{date_suffix}", ckpt_name)
    logger.info("Output directory: %s", output_dir)

    # Determine which trajectories to evaluate
    if traj_indices is not None:
        indices_to_eval = traj_indices
    else:
        indices_to_eval = list(range(min(num_trajectories, n_trajs)))

    # 3. Connect and run
    uri = f"ws://{host}:{port}"
    logger.info("Connecting to %s ...", uri)

    async with websockets.connect(
        uri,
        max_size=None,
        ping_interval=PING_INTERVAL_SECS,
        ping_timeout=PING_TIMEOUT_SECS,
    ) as websocket:
        metadata_raw = await websocket.recv()
        metadata = msgpack_numpy.unpackb(metadata_raw)
        logger.info("Connected. Server metadata: %s", metadata)

        packer = msgpack_numpy.Packer()

        for traj_index in indices_to_eval:
            traj_id = dataset.trajectory_ids[traj_index]
            logger.info("\n{'='*60}\nTrajectory %d (id=%s)\n{'='*60}", traj_index, traj_id)

            if server_type == "dreamdojo":
                await run_trajectory_dreamdojo(
                    websocket,
                    packer,
                    dataset,
                    traj_id,
                    traj_index,
                    action_horizon,
                    embodiment_tag,
                    num_frames,
                    num_frame_per_block,
                )
            else:
                await run_trajectory(
                    websocket,
                    packer,
                    dataset,
                    traj_id,
                    traj_index,
                    action_horizon,
                    embodiment_tag,
                    output_dir if save_gt_video else None,
                    num_frames,
                    num_frame_per_block,
                )

    logger.info("All %d trajectories completed.", len(indices_to_eval))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DreamZero offline eval client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint (for loading modality configs)")
    parser.add_argument("--embodiment-tag", type=str, default="agibot",
                        help="Embodiment tag (agibot, oxe_droid, gr1_unified, ...)")
    parser.add_argument("--video-backend", type=str, default="torchvision_av",
                        help="Video backend (torchvision_av, torchcodec, ffmpeg, pyav)")
    parser.add_argument("--action-horizon", type=int, default=48,
                        help="Steps between inference calls")
    parser.add_argument("--num-trajectories", type=int, default=1,
                        help="Number of trajectories to evaluate (ignored if --traj-indices set)")
    parser.add_argument("--traj-indices", type=str, default=None,
                        help="Comma-separated trajectory indices, e.g. '0,3,7'")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to checkpoint parent dir)")
    parser.add_argument("--language-key", type=str, default=None,
                        help="Override the language modality key, e.g. 'annotation.human.coarse_action'")
    parser.add_argument("--server-type", type=str, default="groot",
                        choices=["groot", "dreamdojo"],
                        help="Server type: 'groot' for step-by-step, 'dreamdojo' for one-shot generation")
    parser.add_argument("--no-save-gt-video", action="store_true", default=False,
                        help="Disable GT video saving (useful when GT already saved from previous run)")

    args = parser.parse_args()

    traj_indices = None
    if args.traj_indices:
        traj_indices = [int(x.strip()) for x in args.traj_indices.split(",")]

    try:
        asyncio.run(eval_client(
            host=args.host,
            port=args.port,
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,

            video_backend=args.video_backend,
            action_horizon=args.action_horizon,
            num_trajectories=args.num_trajectories,
            traj_indices=traj_indices,
            output_dir=args.output_dir,
            language_key=args.language_key,
            server_type=args.server_type,
            save_gt_video=not args.no_save_gt_video,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
