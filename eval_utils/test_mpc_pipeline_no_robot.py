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
import imageio
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


def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
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


def _read_frame_sequence_or_random(
    path: str | None,
    height: int,
    width: int,
    num_frames: int,
    start_frame: int,
) -> np.ndarray:
    """Read a contiguous frame window from mp4; fallback to random RGB frames."""
    frames: list[np.ndarray] = []
    if path and os.path.exists(path):
        try:
            for i, frame in enumerate(iio.imiter(path)):
                if i < start_frame:
                    continue
                frames.append(_resize_frame(np.asarray(frame), height, width))
                if len(frames) >= num_frames:
                    break
        except Exception:
            frames = []

    if not frames:
        base = _read_first_frame_or_random(path, height, width)
        frames = [base]

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:num_frames], axis=0)


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


def _save_candidate_videos(
    candidates: list[dict[str, Any]],
    output_dir: str,
    fps: int,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: list[str] = []
    for i, c in enumerate(candidates):
        frames = c["video_frames"]  # (T, H, W, 3) uint8
        out_path = os.path.join(output_dir, f"candidate_{i}.mp4")
        imageio.mimsave(out_path, frames, fps=fps, codec="libx264")
        saved_paths.append(out_path)
    return saved_paths


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
    parser.add_argument("--save-candidate-videos", action="store_true")
    parser.add_argument("--video-output-dir", type=str, default="debug_outputs/mpc_smoke")
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--warmup-infer-calls", type=int, default=1)
    args = parser.parse_args()

    print(f"Connecting to DreamZero ws://{args.dreamzero_host}:{args.dreamzero_port} ...")
    client = WebsocketClientPolicy(host=args.dreamzero_host, port=args.dreamzero_port)

    left_seq = _read_frame_sequence_or_random(
        args.left_mp4, args.image_h, args.image_w, args.context_frames, args.start_frame
    )
    right_seq = _read_frame_sequence_or_random(
        args.right_mp4, args.image_h, args.image_w, args.context_frames, args.start_frame
    )
    wrist_seq = _read_frame_sequence_or_random(
        args.wrist_mp4, args.image_h, args.image_w, args.context_frames, args.start_frame
    )
    print(
        "Loaded context windows:",
        f"left={tuple(left_seq.shape)} right={tuple(right_seq.shape)} wrist={tuple(wrist_seq.shape)}",
    )

    session_id = str(uuid.uuid4())
    if args.warmup_infer_calls > 0:
        print(f"Running {args.warmup_infer_calls} warmup infer() call(s) to fill temporal buffers...")
        for i in range(args.warmup_infer_calls):
            idx = min(i, args.context_frames - 1)
            warm_obs = {
                "observation/exterior_image_0_left": left_seq[idx],
                "observation/exterior_image_1_left": right_seq[idx],
                "observation/wrist_image_left": wrist_seq[idx],
                "observation/joint_position": np.zeros((7,), dtype=np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": np.zeros((1,), dtype=np.float64),
                "prompt": args.task,
                "session_id": session_id,
            }
            _ = client.infer(warm_obs)

    obs = {
        "observation/exterior_image_0_left": left_seq,
        "observation/exterior_image_1_left": right_seq,
        "observation/wrist_image_left": wrist_seq,
        "observation/joint_position": np.zeros((7,), dtype=np.float64),
        "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
        "observation/gripper_position": np.zeros((1,), dtype=np.float64),
        "prompt": args.task,
        "session_id": session_id,
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

    if args.save_candidate_videos:
        print(f"Saving candidate videos to: {args.video_output_dir}")
        saved_paths = _save_candidate_videos(
            candidates=candidates,
            output_dir=args.video_output_dir,
            fps=args.video_fps,
        )
        for p in saved_paths:
            print(f"  saved: {p}")

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
