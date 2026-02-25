#!/usr/bin/env python
"""
Minimal orchestrator for robots/droid_dreamzero_client.py.

This script is the missing "driver" loop for real-robot DreamZero runs:
1) Connect to droid_dreamzero_client TCP server
2) Send RESET with prompt
3) Repeatedly send RUN_CHUNK
4) Stop on done/truncated or max chunk limit
5) Send CLOSE and exit

Usage:
python robots/droid_dreamzero_orchestrator.py \
    --robot-host 127.0.0.1 \
    --robot-port 6000 \
    --prompt "pick up the red cup" \
    --max-chunks 30
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Any


def _load_remote_utils():
    """Import shared socket helpers from remote_server_utils.py."""
    import importlib.util
    import os

    utils_path = os.path.join(os.path.dirname(__file__), "remote_server_utils.py")
    spec = importlib.util.spec_from_file_location("remote_server_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load remote_server_utils from {utils_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _expect_dict(msg: Any, context: str) -> dict:
    if msg is None:
        raise RuntimeError(f"Connection closed while waiting for {context}")
    if not isinstance(msg, dict):
        raise RuntimeError(f"Expected dict for {context}, got: {type(msg)}")
    return msg


def run_episode(
    send_msg,
    recv_msg,
    sock: socket.socket,
    prompt: str,
    max_chunks: int,
    chunk_interval_seconds: float,
    verbose: bool,
) -> int:
    # RESET
    reset_msg = {"type": "RESET", "prompt": prompt}
    send_msg(sock, reset_msg)
    reset_resp = _expect_dict(recv_msg(sock), "RESET response")
    if reset_resp.get("type") == "ERROR":
        raise RuntimeError(f"RESET failed: {reset_resp.get('message')}")
    if reset_resp.get("type") != "SCENE_DATA":
        raise RuntimeError(f"Unexpected RESET response type: {reset_resp.get('type')}")

    if verbose:
        print("Received initial SCENE_DATA")

    # RUN_CHUNK loop
    chunk_idx = 0
    while chunk_idx < max_chunks:
        send_msg(sock, {"type": "RUN_CHUNK"})
        resp = _expect_dict(recv_msg(sock), "RUN_CHUNK response")
        resp_type = resp.get("type")

        if resp_type == "ERROR":
            raise RuntimeError(f"RUN_CHUNK error: {resp.get('message')}")
        if resp_type != "CHUNK_RESULT":
            raise RuntimeError(f"Unexpected RUN_CHUNK response type: {resp_type}")

        success = bool(resp.get("success", False))
        done = bool(resp.get("done", False))
        truncated = bool(resp.get("truncated", False))
        num_steps = int(resp.get("num_steps", -1))
        print(
            f"[chunk {chunk_idx:03d}] steps={num_steps:02d} "
            f"done={done} truncated={truncated} success={success}"
        )

        chunk_idx += 1

        if done or truncated:
            print("Episode ended by server status.")
            break

        if chunk_interval_seconds > 0:
            time.sleep(chunk_interval_seconds)

    return chunk_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrator for robots/droid_dreamzero_client.py"
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default="127.0.0.1",
        help="Robot laptop host/IP where droid_dreamzero_client is listening",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=6000,
        help="Robot laptop TCP port for droid_dreamzero_client (default: 6000)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Task instruction passed in RESET",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=40,
        help="Maximum RUN_CHUNK calls per episode (default: 40)",
    )
    parser.add_argument(
        "--chunk-interval-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between RUN_CHUNK calls (default: 0)",
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=30.0,
        help="Socket connect timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-close",
        action="store_true",
        help="Do not send CLOSE on exit (useful for debugging)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    utils = _load_remote_utils()
    send_msg = utils.send_msg
    recv_msg = utils.recv_msg

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.connect_timeout_seconds)

    try:
        print(f"Connecting to robot server at {args.robot_host}:{args.robot_port}...")
        sock.connect((args.robot_host, args.robot_port))
        sock.settimeout(None)  # use blocking I/O for episode stream
        print("Connected.")

        chunks_run = run_episode(
            send_msg=send_msg,
            recv_msg=recv_msg,
            sock=sock,
            prompt=args.prompt,
            max_chunks=args.max_chunks,
            chunk_interval_seconds=args.chunk_interval_seconds,
            verbose=args.verbose,
        )
        print(f"Finished episode after {chunks_run} chunk(s).")

        if not args.no_close:
            send_msg(sock, {"type": "CLOSE"})
            if args.verbose:
                print("Sent CLOSE.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if not args.no_close:
            try:
                send_msg(sock, {"type": "CLOSE"})
            except Exception:
                pass
    except Exception as exc:
        print(f"Orchestrator error: {exc}", file=sys.stderr)
        raise
    finally:
        sock.close()
        print("Socket closed.")


if __name__ == "__main__":
    main()
