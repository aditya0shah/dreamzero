"""Client for communicating with a policy server.

Adapted from https://github.com/robo-arena/roboarena/

"""

import io
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image as PILImage

import websockets.sync.client
from typing_extensions import override

from openpi_client.base_policy import BasePolicy
from openpi_client import msgpack_numpy

# The websockets library by default sends a ping every 20 seconds and
# expects a pong response within 20 seconds. However, the sever may not
# send a pong response immediately if it is busy processing a request.
# Increase the ping interval and timeout so that the client can wait
# for a longer time before closing the connection.
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600

class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn = websockets.sync.client.connect(
                self._uri, 
                compression=None, 
                max_size=None,
                ping_interval=PING_INTERVAL_SECS,
                ping_timeout=PING_TIMEOUT_SECS,
            )
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except:
            logging.info("Connection to server with ws:// failed. Trying wss:// ...")
            
        self._uri = "wss://" + self._uri.split("//")[1]
        conn = websockets.sync.client.connect(
            self._uri, 
            compression=None, 
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # Notify server that we're calling the infer endpoint (as opposed to the reset endpoint)
        obs["endpoint"] = "infer"

        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self, reset_info: Dict) -> None:
        # Notify server that we're calling the reset endpoint (as opposed to the infer endpoint)
        reset_info["endpoint"] = "reset"

        data = self._packer.pack(reset_info)
        self._ws.send(data)
        response = self._ws.recv()
        return response

    def plan(self, obs: Dict, seeds: List[int]) -> List[Dict]:
        """Generate N candidate trajectories with different seeds (MPC planning).

        Args:
            obs: Observation dict (same format as infer()).
            seeds: List of N integer seeds for diverse generation.

        Returns:
            List of N dicts, each containing:
              - "action.joint_position": np.ndarray (T, 7)
              - "action.gripper_position": np.ndarray (T, 1)
              - "video_frames": np.ndarray (T, H, W, 3) uint8 RGB
        """
        obs["endpoint"] = "plan"
        obs["_plan_seeds"] = seeds

        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in plan():\n{response}")

        result = msgpack_numpy.unpackb(response)
        candidates = result["candidates"]

        # Decode JPEG frames back to numpy arrays
        decoded_candidates = []
        for c in candidates:
            jpeg_frames = c["video_frames_jpeg"]
            shape = tuple(c["video_frames_shape"])
            frames = np.stack([
                np.array(PILImage.open(io.BytesIO(jpg_bytes)))
                for jpg_bytes in jpeg_frames
            ])
            decoded_candidates.append({
                "action.joint_position": c["action.joint_position"],
                "action.gripper_position": c["action.gripper_position"],
                "video_frames": frames,
            })

        return decoded_candidates

    def commit(self, best_idx: int) -> None:
        """Tell the server to adopt the KV cache state from candidate best_idx.

        Must be called after plan(). After commit, subsequent infer() calls
        build on the chosen trajectory's context.

        Args:
            best_idx: Index of the chosen candidate (0-based).
        """
        msg = {
            "endpoint": "commit",
            "_commit_idx": best_idx,
        }
        data = self._packer.pack(msg)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, bytes):
            # Unexpected bytes response
            raise RuntimeError(f"Unexpected response from commit(): {response}")
        # Expected: "commit successful" string


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WebsocketClientPolicy()
    actions = client.infer({})
    print(f"Actions received: {actions}")
    client.reset({})