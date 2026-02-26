#!/usr/bin/env python
"""
DROID robot server that runs DreamZero inference internally.

This server implements a chunk-based control protocol:
1. Client sends RESET + prompt  → Server resets robot, inits DreamZero session, returns scene
2. Client sends RUN_CHUNK       → Server calls DreamZero, executes action chunk, returns next scene
3. Loop continues until episode ends or client sends CLOSE

Compared to droid_trajectory_server.py (Ctrl-World):
- Trajectories are generated internally via DreamZero WebSocket (not sent by client)
- Action space: joint_position (not joint_velocity)
- Image resolution: 320x180 (DreamZero-DROID training resolution)
- Frame buffer maintained across chunks with subsampled capture during execution

Prerequisites:
1. DreamZero inference server running on compute node:
     python -m torch.distributed.run --standalone --nproc_per_node=2 \\
       socket_test_optimized_AR.py --port 8000 --enable-dit-cache --model-path <ckpt>
2. DROID package installed: https://github.com/droid-dataset/droid
3. openpi_client installed: pip install openpi-client

Example usage:
# Terminal 1: Start this server on the robot
python robots/droid_dreamzero_client.py \\
    --left-camera-id "24259877" \\
    --right-camera-id "24514023" \\
    --wrist-camera-id "13062452" \\
    --dreamzero-host <compute-node-ip> \\
    --dreamzero-port 8000 \\
    --server-port 6000

# Terminal 2: Orchestrator connects and sends commands
# RESET {"type": "RESET", "prompt": "pick up the red cup"} → SCENE_DATA
# RUN_CHUNK {"type": "RUN_CHUNK"} → CHUNK_RESULT (with next_scene)
# CLOSE → closes connection

Keyboard Controls (in Terminal 1):
  's' - Mark current episode as SUCCESS
  'f' - Mark current episode as FAILURE
  'q' - Quit server
"""

import argparse
import io
import json
import os
import sys
import time
import threading
import uuid
import importlib.util
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Import shared utilities from remote_server_utils.py (same dir as this file,
# or from the directory where droid_trajectory_server.py lives).
# Adjust the path below if remote_server_utils.py lives elsewhere.
# ---------------------------------------------------------------------------
_utils_candidates = [
    os.path.join(os.path.dirname(__file__), "remote_server_utils.py"),
    os.path.join(os.path.dirname(__file__), "..", "remote_server_utils.py"),
]
_utils_path = next((p for p in _utils_candidates if os.path.exists(p)), None)

if _utils_path is None:
    raise FileNotFoundError(
        "Could not find remote_server_utils.py. "
        "Place it alongside this script or update _utils_candidates."
    )

spec = importlib.util.spec_from_file_location("remote_server_utils", _utils_path)
remote_server_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_server_utils)

EpisodeState = remote_server_utils.EpisodeState
keyboard_listener = remote_server_utils.keyboard_listener
send_msg = remote_server_utils.send_msg
recv_msg = remote_server_utils.recv_msg

# ---------------------------------------------------------------------------
# DreamZero WebSocket client (from eval_utils/policy_client.py)
# ---------------------------------------------------------------------------
# Add DreamZero repo root to path so we can import eval_utils
_dreamzero_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dreamzero_root not in sys.path:
    sys.path.insert(0, _dreamzero_root)

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# DROID imports
# ---------------------------------------------------------------------------
try:
    from droid.robot_env import RobotEnv
    from PIL import Image
    DROID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DROID not found: {e}")
    DROID_AVAILABLE = False

# ---------------------------------------------------------------------------
# Reward FM client for MPC scoring
# ---------------------------------------------------------------------------

class RewardClient:
    """Client for the Reward FM evaluation server (eval_server.py).

    Sends imagined video frames to the reward model and returns a progress
    score in [0, 1].
    """

    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

    def score_trajectories(
        self,
        candidates: List[Dict[str, Any]],
        task: str,
    ) -> List[float]:
        """Score multiple candidate trajectories in a single batch request.

        Args:
            candidates: List of dicts with "video_frames" key (T, H, W, 3) uint8.
            task: Language task instruction.

        Returns:
            List of progress scores (one per candidate), each in [0, 1].
        """
        # Reward server expects ndarray frames via /evaluate_batch_npy multipart payload.
        files = {}
        data = {}
        for i, c in enumerate(candidates):
            frames = c["video_frames"]  # (T, H, W, 3) uint8
            if not isinstance(frames, np.ndarray):
                frames = np.array(frames, dtype=np.uint8)

            file_key = f"sample_{i}_trajectory_frames"
            buf = io.BytesIO()
            np.save(buf, frames)
            buf.seek(0)
            files[file_key] = (f"{file_key}.npy", buf, "application/octet-stream")

            sample = {
                "sample_type": "progress",
                "trajectory": {
                    "frames": {"__numpy_file__": file_key},
                    "task": task,
                    "frames_shape": [int(x) for x in frames.shape],
                    "id": f"plan_candidate_{i}",
                },
            }
            data[f"sample_{i}"] = json.dumps(sample)

        resp = requests.post(
            f"{self.base_url}/evaluate_batch_npy",
            files=files,
            data=data,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        progress_preds = result["outputs_progress"]["progress_pred"]
        scores = []
        for pred in progress_preds:
            # pred is a list of per-frame progress values; take the last one
            if isinstance(pred, list) and len(pred) > 0:
                scores.append(float(pred[-1]))
            else:
                scores.append(0.0)
        return scores


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DROID_CONTROL_HZ = 15

# DreamZero-DROID training resolution
IMAGE_H = 180
IMAGE_W = 320

# Frame schedule (matches test_client_AR.py)
RELATIVE_OFFSETS = [-23, -16, -8, 0]

# Within each executed chunk, capture frames at these step indices
# (0, 7, 15, 23 gives 4 frames spread across 24 steps ≈ 1.5 s window)
CAPTURE_STEPS = {0, 7, 15, 23}

# DreamZero observation keys → which camera ID maps to which key
# Populated at runtime once camera IDs are known.
CAM_OBS_KEYS = [
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
]


# ---------------------------------------------------------------------------
# Robot init
# ---------------------------------------------------------------------------

def init_robot(left_camera_id: str, right_camera_id: str, wrist_camera_id: str):
    if not DROID_AVAILABLE:
        raise RuntimeError("DROID not installed.")

    print("Connecting to DROID robot...")
    # joint_position action space — DreamZero outputs joint positions
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("✓ DROID robot connected")

    camera_config = {
        "left_camera_id": left_camera_id,
        "right_camera_id": right_camera_id,
        "wrist_camera_id": wrist_camera_id,
    }
    return env, camera_config


# ---------------------------------------------------------------------------
# Scene / image extraction
# ---------------------------------------------------------------------------

def extract_scene_data(
    env_obs: Dict[str, Any],
    camera_config: Dict[str, str],
    target_width: int = IMAGE_W,
    target_height: int = IMAGE_H,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Extract and resize camera images + robot state from a DROID observation.

    Returns:
        images: dict with keys "left_image", "right_image", "wrist_image"
                each (target_height, target_width, 3) uint8 RGB
        metadata: dict with cartesian_position, joint_positions, gripper_position
    """
    image_observations = env_obs["image"]
    left_image = right_image = wrist_image = None

    for key in image_observations:
        if camera_config["left_camera_id"] in key and "left" in key:
            left_image = image_observations[key]
        elif camera_config["right_camera_id"] in key and "left" in key:
            right_image = image_observations[key]
        elif camera_config["wrist_camera_id"] in key and "left" in key:
            wrist_image = image_observations[key]

    for name, img in [("left", left_image), ("right", right_image), ("wrist", wrist_image)]:
        if img is None:
            raise ValueError(f"Could not find {name} camera image.")

    def _process(img):
        img = img[..., :3]          # drop alpha
        img = img[..., ::-1]        # BGR → RGB
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.resize((target_width, target_height), Image.LANCZOS)
        return np.array(pil)

    images = {
        "left_image":  _process(left_image),
        "right_image": _process(right_image),
        "wrist_image": _process(wrist_image),
    }

    robot_state = env_obs["robot_state"]
    metadata = {
        "cartesian_position": np.array(robot_state["cartesian_position"]).tolist(),
        "joint_positions":    np.array(robot_state["joint_positions"]).tolist(),
        "gripper_position":   float(robot_state["gripper_position"]),
    }

    return images, metadata


def images_to_buffer(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map extract_scene_data image keys → DreamZero observation keys."""
    return {
        "observation/exterior_image_0_left": images["left_image"],
        "observation/exterior_image_1_left": images["right_image"],
        "observation/wrist_image_left":      images["wrist_image"],
    }


# ---------------------------------------------------------------------------
# Observation construction for DreamZero
# ---------------------------------------------------------------------------

def build_obs(
    frame_buffer: Dict[str, list],
    indices: list,
    single_frame: bool,
    prompt: str,
    session_id: str,
    robot_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a DreamZero observation dict.

    Args:
        frame_buffer: per-key lists of (H, W, 3) uint8 frames
        indices: which frame indices to select
        single_frame: if True, images are (H, W, 3); else (T, H, W, 3)
        prompt: language instruction string
        session_id: UUID string for this episode
        robot_state: raw robot_state dict from env_obs["robot_state"]
    """
    obs: Dict[str, Any] = {}

    for key in CAM_OBS_KEYS:
        frames_list = frame_buffer[key]
        selected = np.stack([frames_list[i] for i in indices])  # (T, H, W, 3)
        if single_frame:
            selected = selected[0]                              # (H, W, 3)
        obs[key] = selected.astype(np.uint8)

    obs["observation/joint_position"]     = np.array(robot_state["joint_positions"],    dtype=np.float64)
    obs["observation/cartesian_position"] = np.array(robot_state["cartesian_position"], dtype=np.float64)
    obs["observation/gripper_position"]   = np.array([robot_state["gripper_position"]], dtype=np.float64)
    obs["prompt"]     = prompt
    obs["session_id"] = session_id
    return obs


# ---------------------------------------------------------------------------
# Chunk execution
# ---------------------------------------------------------------------------

def execute_chunk(
    env: "RobotEnv",
    actions_dict: Dict[str, np.ndarray],
    frame_buffer: Dict[str, list],
    camera_config: Dict[str, str],
    episode_state: "EpisodeState",
    max_steps: Optional[int],
) -> Dict[str, Any]:
    """Execute a DreamZero action chunk on the robot.

    Captures frames at CAPTURE_STEPS within the chunk for the next inference call.

    Args:
        actions_dict: dict with "action.joint_position" (T,7) and
                      "action.gripper_position" (T,1) from DreamZero
        frame_buffer: per-key frame lists — updated in-place
    """
    joint_pos = actions_dict["action.joint_position"]    # (T, 7) float32
    gripper   = actions_dict["action.gripper_position"]  # (T, 1) float32

    done = truncated = success = False
    status_info = {}
    num_steps_executed = 0

    for i, (jpos, grip) in enumerate(zip(joint_pos, gripper)):
        step_start = time.time()

        # Capture frame at subsampled steps for the next inference call
        if i in CAPTURE_STEPS:
            raw_obs = env.get_observation()
            img, _ = extract_scene_data(raw_obs, camera_config)
            for key, frame in images_to_buffer(img).items():
                frame_buffer[key].append(frame)

        # Build action: [joint_pos(7), gripper(1)]
        action = np.concatenate([np.array(jpos, dtype=np.float32),
                                  np.array(grip, dtype=np.float32).flatten()])

        # Binarize gripper
        action[-1] = 1.0 if action[-1] > 0.5 else 0.0

        try:
            env.step(action)
        except Exception as e:
            return {
                "success": False, "done": False, "truncated": False,
                "num_steps": num_steps_executed,
                "error": f"env.step failed at step {i}: {e}",
            }

        episode_state.increment_step()
        num_steps_executed += 1
        current_step = episode_state.get_step_count()

        done, truncated, success, status_info = episode_state.get_status()

        if max_steps and current_step >= max_steps:
            truncated = True
            status_info["timeout"] = True

        is_last = (i == len(joint_pos) - 1)
        status_str = " - SUCCESS" if success else (" - FAILURE" if done else "")
        print(f"  Step {current_step} ({i+1}/{len(joint_pos)}){status_str}", end="\r")

        if done or truncated:
            print()
            break
        if is_last:
            print()

        # Maintain control frequency
        elapsed = time.time() - step_start
        sleep_time = 1.0 / DROID_CONTROL_HZ - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Handle timeout confirmation
    if truncated and status_info.get("timeout", False):
        print("\n" + "─" * 60)
        print("⏱️  TIMEOUT — press 's' for SUCCESS or 'f' for FAILURE")
        print("─" * 60)
        while episode_state.success is None:
            time.sleep(0.1)
        _, _, success, status_info = episode_state.get_status()

    return {
        "success": success,
        "done": done,
        "truncated": truncated,
        "num_steps": num_steps_executed,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Client handler
# ---------------------------------------------------------------------------

def handle_client(conn, addr, env, camera_config, dz_client, args, episode_state,
                  reward_client=None):
    """Handle one orchestrator connection with the RUN_CHUNK loop."""
    print(f"Connected: {addr}")

    # Per-episode state (reset on each RESET command)
    frame_buffer: Dict[str, list] = {k: [] for k in CAM_OBS_KEYS}
    chunk_idx = 0
    action_horizon = 24   # updated dynamically from DreamZero response
    session_id = None
    prompt = ""

    try:
        while True:
            cmd = recv_msg(conn)
            if cmd is None:
                print("Orchestrator disconnected")
                dz_client.reset({})
                break

            # ------------------------------------------------------------------
            if cmd["type"] == "RESET":
                print("\n" + "=" * 60)
                print("RESET — starting new episode")
                print("=" * 60)

                prompt = cmd.get("prompt", "")
                print(f"Prompt: {prompt!r}")

                # Reset episode state and per-episode buffers
                episode_state.reset()
                frame_buffer = {k: [] for k in CAM_OBS_KEYS}
                chunk_idx = 0
                session_id = str(uuid.uuid4())

                # Tell DreamZero server to clear its state
                dz_client.reset({})

                # Reset robot
                env.reset_rewardfm()

                # Capture initial frame (frame 0)
                raw_obs = env.get_observation()
                images, metadata = extract_scene_data(raw_obs, camera_config)
                for key, frame in images_to_buffer(images).items():
                    frame_buffer[key].append(frame)

                print(f"Session ID: {session_id}")
                print(f"Max steps: {args.max_steps or 'unlimited'}\n")

                send_msg(conn, {
                    "type": "SCENE_DATA",
                    "left_image":  images["left_image"],
                    "right_image": images["right_image"],
                    "wrist_image": images["wrist_image"],
                    "metadata":    metadata,
                })

            # ------------------------------------------------------------------
            elif cmd["type"] == "GET_SCENE":
                raw_obs = env.get_observation()
                images, metadata = extract_scene_data(raw_obs, camera_config)
                send_msg(conn, {
                    "type": "SCENE_DATA",
                    "left_image":  images["left_image"],
                    "right_image": images["right_image"],
                    "wrist_image": images["wrist_image"],
                    "metadata":    metadata,
                })

            # ------------------------------------------------------------------
            elif cmd["type"] == "RUN_CHUNK":
                if session_id is None:
                    send_msg(conn, {"type": "ERROR", "message": "Send RESET before RUN_CHUNK"})
                    continue

                # Override prompt if provided in this message
                if "prompt" in cmd:
                    prompt = cmd["prompt"]

                # Build DreamZero observation
                raw_obs = env.get_observation()
                robot_state = raw_obs["robot_state"]

                if chunk_idx == 0:
                    # Step 0: single frame
                    obs = build_obs(frame_buffer, [0], single_frame=True,
                                    prompt=prompt, session_id=session_id,
                                    robot_state=robot_state)
                else:
                    anchor = 23 + (chunk_idx - 1) * action_horizon
                    indices = [max(anchor + off, 0) for off in RELATIVE_OFFSETS]
                    # Clamp to buffer length in case of early episodes
                    buf_len = len(frame_buffer[CAM_OBS_KEYS[0]])
                    indices = [min(idx, buf_len - 1) for idx in indices]
                    obs = build_obs(frame_buffer, indices, single_frame=False,
                                    prompt=prompt, session_id=session_id,
                                    robot_state=robot_state)

                mpc_enabled = args.mpc and reward_client is not None

                print(f"\n[RUN_CHUNK {chunk_idx}] {'MPC' if mpc_enabled else 'Standard'} mode...")
                infer_start = time.time()

                if mpc_enabled:
                    # --- MPC Planning ---
                    n_candidates = args.mpc_n_candidates
                    seeds = list(range(n_candidates))
                    print(f"  Planning with N={n_candidates} candidates...")

                    try:
                        candidates = dz_client.plan(obs, seeds)
                    except Exception as e:
                        send_msg(conn, {"type": "ERROR",
                                        "message": f"DreamZero plan failed: {e}"})
                        continue
                    print(f"  DreamZero plan latency: {time.time() - infer_start:.2f}s")

                    # Score each candidate with the reward model
                    score_start = time.time()
                    try:
                        scores = reward_client.score_trajectories(candidates, prompt)
                    except Exception as e:
                        print(f"  WARNING: Reward scoring failed: {e}")
                        print(f"  Falling back to candidate 0")
                        scores = [0.0] * n_candidates
                        scores[0] = 1.0
                    print(f"  Reward scoring latency: {time.time() - score_start:.2f}s")
                    print(f"  Scores: {[f'{s:.3f}' for s in scores]}")

                    best_idx = int(np.argmax(scores))
                    print(f"  Best candidate: {best_idx} (score={scores[best_idx]:.3f})")

                    actions_dict = {
                        "action.joint_position": candidates[best_idx]["action.joint_position"],
                        "action.gripper_position": candidates[best_idx]["action.gripper_position"],
                    }

                    # Commit the chosen candidate's KV state on DreamZero server
                    try:
                        dz_client.commit(best_idx)
                    except Exception as e:
                        print(f"  WARNING: commit() failed: {e}")
                else:
                    # --- Standard single inference ---
                    try:
                        actions_dict = dz_client.infer(obs)
                    except Exception as e:
                        send_msg(conn, {"type": "ERROR",
                                        "message": f"DreamZero infer failed: {e}"})
                        continue
                    print(f"  DreamZero latency: {time.time() - infer_start:.2f}s")

                # Read action horizon from response
                action_horizon = actions_dict["action.joint_position"].shape[0]
                print(f"  Action horizon: {action_horizon} steps")

                # Execute chunk on robot, capturing frames for next call
                result = execute_chunk(
                    env, actions_dict, frame_buffer, camera_config,
                    episode_state, args.max_steps,
                )
                chunk_idx += 1

                if result["error"]:
                    send_msg(conn, {"type": "ERROR", "message": result["error"]})
                    continue

                # Capture scene after chunk execution
                raw_obs = env.get_observation()
                images, metadata = extract_scene_data(raw_obs, camera_config)

                send_msg(conn, {
                    "type":      "CHUNK_RESULT",
                    "success":   result["success"],
                    "done":      result["done"],
                    "truncated": result["truncated"],
                    "num_steps": result["num_steps"],
                    "next_scene": {
                        "left_image":  images["left_image"],
                        "right_image": images["right_image"],
                        "wrist_image": images["wrist_image"],
                        "metadata":    metadata,
                    },
                })

            # ------------------------------------------------------------------
            elif cmd["type"] == "CLOSE":
                print("CLOSE received — ending connection")
                dz_client.reset({})   # triggers MP4 save on DreamZero server
                break

            else:
                send_msg(conn, {"type": "ERROR", "message": f"Unknown command: {cmd['type']}"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error handling client: {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_server(args):
    import socket as _socket

    env, camera_config = init_robot(
        args.left_camera_id, args.right_camera_id, args.wrist_camera_id
    )

    print(f"Connecting to DreamZero at {args.dreamzero_host}:{args.dreamzero_port}...")
    dz_client = WebsocketClientPolicy(host=args.dreamzero_host, port=args.dreamzero_port)
    print("✓ Connected to DreamZero server")

    # MPC reward client (if enabled)
    reward_client = None
    if args.mpc:
        print(f"MPC mode enabled: N={args.mpc_n_candidates} candidates")
        print(f"Connecting to reward server at {args.reward_server_host}:{args.reward_server_port}...")
        reward_client = RewardClient(
            host=args.reward_server_host, port=args.reward_server_port
        )
        print("✓ Reward client configured")

    episode_state = EpisodeState(max_steps=args.max_steps)
    keyboard_thread = threading.Thread(
        target=keyboard_listener, args=(episode_state,), daemon=True
    )
    keyboard_thread.start()

    server_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    server_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", args.server_port))
    server_sock.listen(5)
    server_sock.settimeout(1.0)

    print(f"\nDROID DreamZero server listening on 0.0.0.0:{args.server_port}")
    print("Waiting for orchestrator connection...\n")

    try:
        while True:
            try:
                conn, addr = server_sock.accept()
                handle_client(conn, addr, env, camera_config, dz_client, args,
                              episode_state, reward_client=reward_client)
            except _socket.timeout:
                continue
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server_sock.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DROID robot server with DreamZero inference"
    )

    # Camera IDs
    parser.add_argument("--left-camera-id",  type=str, required=True)
    parser.add_argument("--right-camera-id", type=str, required=True)
    parser.add_argument("--wrist-camera-id", type=str, required=True)

    # DreamZero server
    parser.add_argument("--dreamzero-host", type=str, default="localhost",
                        help="Hostname/IP of the DreamZero inference server")
    parser.add_argument("--dreamzero-port", type=int, default=8000,
                        help="Port of the DreamZero inference server")

    # This TCP server
    parser.add_argument("--server-port", type=int, default=6000,
                        help="TCP port for orchestrator to connect to")

    # Episode settings
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Max steps per episode before timeout (default: 600)")

    # MPC planning mode
    parser.add_argument("--mpc", action="store_true",
                        help="Enable MPC planning mode (requires reward server)")
    parser.add_argument("--mpc-n-candidates", type=int, default=8,
                        help="Number of candidate trajectories for MPC (default: 8)")
    parser.add_argument("--reward-server-host", type=str, default="localhost",
                        help="Hostname/IP of the Reward FM server")
    parser.add_argument("--reward-server-port", type=int, default=8001,
                        help="Port of the Reward FM server (default: 8001)")

    args = parser.parse_args()

    print("=" * 60)
    print("DROID DreamZero Client")
    print("=" * 60)
    print(f"Left camera:       {args.left_camera_id}")
    print(f"Right camera:      {args.right_camera_id}")
    print(f"Wrist camera:      {args.wrist_camera_id}")
    print(f"DreamZero server:  {args.dreamzero_host}:{args.dreamzero_port}")
    print(f"TCP server port:   {args.server_port}")
    print(f"Image resolution:  {IMAGE_W}x{IMAGE_H} (DreamZero-DROID)")
    print(f"Max steps:         {args.max_steps}")
    if args.mpc:
        print(f"MPC mode:          ON (N={args.mpc_n_candidates})")
        print(f"Reward server:     {args.reward_server_host}:{args.reward_server_port}")
    else:
        print(f"MPC mode:          OFF")
    print("=" * 60)

    run_server(args)


if __name__ == "__main__":
    main()
