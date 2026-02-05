#!/usr/bin/env python3
"""Test client for AR_droid policy server using roboarena interface.

This script tests the AR_droid policy server when running with --roboarena_server flag.

Expected server configuration:
    - image_resolution: (180, 320)
    - n_external_cameras: 2
    - needs_wrist_camera: True
    - action_space: "joint_position"

Usage:
    # Start server with roboarena interface:
    torchrun --nproc_per_node=8 socket_test_optimized_AR_droid.py --roboarena_server --port 8000
    
    # Run this test:
    python test_policy_server_ar_droid.py --host <server_host> --port 8000
"""

import argparse
import logging
import time

import numpy as np

import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy


def _make_ar_droid_observation(
    server_config: policy_server.PolicyServerConfig,
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> dict:
    """Create a dummy observation matching AR_droid expectations.
    
    AR_droid expects:
        - 2 external cameras (exterior_image_0_left, exterior_image_1_left)
        - 1 wrist camera (wrist_image_left)
        - Image resolution: 180x320 (H x W)
        - joint_position: 7 DoF
        - gripper_position: 1 DoF
    """
    obs = {}
    
    # Determine image resolution
    if server_config.image_resolution is not None:
        h, w = server_config.image_resolution
    else:
        # Default for AR_droid
        h, w = 180, 320
    
    # External cameras (0-indexed in roboarena)
    for i in range(server_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Wrist camera
    if server_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Session ID - should be passed in to ensure consistency within a session
    if server_config.needs_session_id:
        import uuid
        # Generate unique session ID if not provided
        obs["session_id"] = session_id if session_id else str(uuid.uuid4())
    
    # State observations (AR_droid: 7 DoF arm + 1 gripper)
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    
    # Language prompt
    obs["prompt"] = prompt
    
    return obs


def test_ar_droid_policy_server(host: str = "localhost", port: int = 8000, num_inferences: int = 5):
    """Test the AR_droid policy server with roboarena interface.
    
    Args:
        host: Server hostname
        port: Server port
        num_inferences: Number of inference calls to make
    """
    logging.info(f"Connecting to AR_droid server at {host}:{port}...")
    
    client = WebsocketClientPolicy(host=host, port=port)
    
    # Validate server metadata
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    assert isinstance(metadata, dict), "Metadata should be a dict"
    
    try:
        server_config = policy_server.PolicyServerConfig(**metadata)
    except Exception as e:
        logging.error(f"Error parsing metadata: {e}")
        raise e
    
    # Validate expected AR_droid configuration
    logging.info(f"Server config: {server_config}")
    assert server_config.n_external_cameras == 2, f"Expected 2 external cameras, got {server_config.n_external_cameras}"
    assert server_config.needs_wrist_camera, "Expected wrist camera to be enabled"
    assert server_config.action_space == "joint_position", f"Expected joint_position action space, got {server_config.action_space}"
    
    logging.info("Server configuration validated for AR_droid")
    
    # Generate unique session ID for this test run
    import uuid
    session_id = str(uuid.uuid4())
    logging.info(f"Generated session ID: {session_id}")
    
    # Test multiple inference calls (to test frame accumulation)
    prompts = [
        "pick up the red block",
        "pick up the red block",
        "pick up the red block",
        "pick up the red block",
        "pick up the red block",
        # "place the object on the table",
        # "open the drawer",
        # "close the gripper",
        # "move to home position",
    ]
    
    for i in range(num_inferences):
        prompt = prompts[i % len(prompts)]
        obs = _make_ar_droid_observation(server_config, prompt=prompt, session_id=session_id)
        
        logging.info(f"Inference {i+1}/{num_inferences}: prompt='{prompt}'")
        start_time = time.time()
        
        actions = client.infer(obs)
        
        elapsed = time.time() - start_time
        logging.info(f"  Response received in {elapsed:.2f}s")
        
        # Validate action format
        assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
        assert len(actions.shape) == 2, f"Expected 2D array, got shape {actions.shape}"
        
        # AR_droid with joint_position action space should return (N, 8): 7 joints + 1 gripper
        assert actions.shape[-1] == 8, f"Expected 8 action dimensions (7 joints + 1 gripper), got {actions.shape[-1]}"
        
        logging.info(f"  Action shape: {actions.shape}, dtype: {actions.dtype}")
        logging.info(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    
    # Test reset functionality
    logging.info("Testing reset...")
    client.reset({})
    logging.info("Reset successful")
    
    # Test one more inference after reset with a new session ID
    # This tests that session change detection works properly
    new_session_id = str(uuid.uuid4())
    logging.info(f"Testing inference after reset with new session ID: {new_session_id}")
    obs = _make_ar_droid_observation(server_config, prompt="test after reset", session_id=new_session_id)
    actions = client.infer(obs)
    assert isinstance(actions, np.ndarray), "Post-reset inference failed"
    logging.info(f"Post-reset action shape: {actions.shape}")
    
    logging.info("=" * 60)
    logging.info("All tests passed!")
    logging.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test AR_droid policy server with roboarena interface"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--num-inferences",
        type=int,
        default=5,
        help="Number of inference calls to make (default: 5)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    test_ar_droid_policy_server(
        host=args.host,
        port=args.port,
        num_inferences=args.num_inferences,
    )


if __name__ == "__main__":
    main()
