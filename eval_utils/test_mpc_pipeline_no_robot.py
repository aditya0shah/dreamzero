#!/usr/bin/env python
"""
No-robot smoke test for DreamZero MPC pipeline.

This script validates:
1) DreamZero websocket connectivity
2) plan(obs, seeds) returns multiple candidates
3) commit(best_idx) works
4) (optional) Reward server scoring via /evaluate_batch

Input observations come from prerecorded MP4s if available, or random images.
"""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

import imageio.v3 as iio
import numpy as np
import requests

from eval_utils.policy_client import WebsocketClientPolicy


def _read_first_frame_or_random(path: str | None, height: int, width: int) -> np.ndarray:
    """Read first frame from mp4 path; fallback to random uint8 RGB."""
    if path and os.path.exists(path):
        frame = iio.imread(path, index=0)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[0] != height or frame.shape[1] != width:
            # Avoid extra deps (cv2/PIL). Nearest-neighbor by indexing.
            y_idx = (np.linspace(0, frame.shape[0] - 1, height)).astype(np.int64)
            x_idx = (np.linspace(0, frame.shape[1] - 1, width)).astype(np.int64)
            frame = frame[y_idx][:, x_idx]
        return frame

    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _score_with_reward_server(
    reward_url: str,
    candidates: list[dict[str, Any]],
    task: str,
    timeout_seconds: int,
) -> list[float]:
    samples = []
    for i, c in enumerate(candidates):
        frames = c["video_frames"]  # (T, H, W, 3) uint8
        samples.append(
            {
                "sample_type": "progress",
                "trajectory": {
                    "frames": frames.tolist(),
                    "task": task,
                    "frames_shape": list(frames.shape),
                    "id": f"smoke_candidate_{i}",
                },
            }
        )
    payload = {"samples": samples}
    resp = requests.post(reward_url, json=payload, timeout=timeout_seconds)
    resp.raise_for_status()
    out = resp.json()
    preds = out["outputs_progress"]["progress_pred"]
    scores = []
    for pred in preds:
        if isinstance(pred, list) and pred:
            scores.append(float(pred[-1]))
        else:
            scores.append(0.0)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="No-robot DreamZero MPC smoke test")
    parser.add_argument("--dreamzero-host", type=str, default="127.0.0.1")
    parser.add_argument("--dreamzero-port", type=int, default=8000)
    parser.add_argument("--task", type=str, default="pick up the red cup")
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--image-h", type=int, default=180)
    parser.add_argument("--image-w", type=int, default=320)
    parser.add_argument(
        "--left-mp4",
        type=str,
        default="/home/adityashah/Documents/dreamzero/debug_image/exterior_image_1_left.mp4",
    )
    parser.add_argument(
        "--right-mp4",
        type=str,
        default="/home/adityashah/Documents/dreamzero/debug_image/exterior_image_2_left.mp4",
    )
    parser.add_argument(
        "--wrist-mp4",
        type=str,
        default="/home/adityashah/Documents/dreamzero/debug_image/wrist_image_left.mp4",
    )
    parser.add_argument("--reward-host", type=str, default="")
    parser.add_argument("--reward-port", type=int, default=8001)
    parser.add_argument("--reward-https", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    print(f"Connecting to DreamZero ws://{args.dreamzero_host}:{args.dreamzero_port} ...")
    client = WebsocketClientPolicy(host=args.dreamzero_host, port=args.dreamzero_port)

    left = _read_first_frame_or_random(args.left_mp4, args.image_h, args.image_w)
    right = _read_first_frame_or_random(args.right_mp4, args.image_h, args.image_w)
    wrist = _read_first_frame_or_random(args.wrist_mp4, args.image_h, args.image_w)

    obs = {
        "observation/exterior_image_0_left": left,
        "observation/exterior_image_1_left": right,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": np.zeros((7,), dtype=np.float64),
        "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
        "observation/gripper_position": np.zeros((1,), dtype=np.float64),
        "prompt": args.task,
        "session_id": str(uuid.uuid4()),
    }

    print("Resetting DreamZero session...")
    client.reset({})

    seeds = list(range(args.n_candidates))
    print(f"Requesting plan() with seeds={seeds} ...")
    candidates = client.plan(obs, seeds)
    print(f"Received {len(candidates)} candidates")
    for i, c in enumerate(candidates):
        j = c["action.joint_position"]
        g = c["action.gripper_position"]
        v = c["video_frames"]
        print(
            f"  candidate[{i}]: "
            f"joint={tuple(j.shape)} gripper={tuple(g.shape)} video={tuple(v.shape)} dtype={v.dtype}"
        )

    if not candidates:
        raise RuntimeError("No candidates returned from DreamZero plan()")

    best_idx = 0
    if args.reward_host:
        scheme = "https" if args.reward_https else "http"
        reward_url = f"{scheme}://{args.reward_host}:{args.reward_port}/evaluate_batch"
        print(f"Scoring with reward server: {reward_url}")
        scores = _score_with_reward_server(
            reward_url=reward_url,
            candidates=candidates,
            task=args.task,
            timeout_seconds=args.timeout_seconds,
        )
        print("Scores:", [f"{s:.4f}" for s in scores])
        best_idx = int(np.argmax(scores))
        print(f"Best idx from reward: {best_idx}")

    print(f"Committing candidate {best_idx} ...")
    client.commit(best_idx)
    print("Commit successful. Smoke test passed.")


if __name__ == "__main__":
    main()
