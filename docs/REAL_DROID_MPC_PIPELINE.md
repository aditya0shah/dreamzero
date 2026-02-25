# Real DROID MPC Pipeline (DreamZero + Reward FM)

This runbook describes the full 3-machine setup:

- Robot laptop: `robots/droid_dreamzero_client.py` (robot control + MPC coordinator)
- H100 cluster: `socket_test_optimized_AR.py` (DreamZero world model server)
- PC: `reward_fm/rfm/evals/eval_server.py` (trajectory scorer)

It also includes the orchestrator client:

- Any machine: `robots/droid_dreamzero_orchestrator.py` (sends `RESET` / `RUN_CHUNK` / `CLOSE`)

---

## 1) Start DreamZero server (H100 cluster)

```bash
cd /home/adityashah/Documents/dreamzero
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 8000 \
  --enable-dit-cache \
  --model-path <DREAMZERO_CHECKPOINT_PATH>
```

Expected: server listens on `0.0.0.0:8000` and accepts websocket clients.

---

## 2) Start Reward server (PC)

```bash
cd /home/adityashah/Documents/reward_fm
python rfm/evals/eval_server.py \
  model_path=<RFM_MODEL_PATH> \
  num_gpus=1 \
  server_port=8001
```

Expected: HTTP server listens on `0.0.0.0:8001` with `POST /evaluate_batch`.

---

## 3) Start robot-side DreamZero server (Robot laptop)

```bash
cd /home/adityashah/Documents/dreamzero
python robots/droid_dreamzero_client.py \
  --left-camera-id "<LEFT_CAMERA_ID>" \
  --right-camera-id "<RIGHT_CAMERA_ID>" \
  --wrist-camera-id "<WRIST_CAMERA_ID>" \
  --dreamzero-host "<H100_HOST_OR_IP>" \
  --dreamzero-port 8000 \
  --server-port 6000 \
  --mpc \
  --mpc-n-candidates 8 \
  --reward-server-host "<PC_HOST_OR_IP>" \
  --reward-server-port 8001
```

Expected logs:

- `Connected to DreamZero server`
- `Reward client configured`
- `DROID DreamZero server listening on 0.0.0.0:6000`

---

## 4) Start orchestrator (any machine)

`droid_dreamzero_client.py` is a TCP server. You still need a small client to drive episodes.

```bash
cd /home/adityashah/Documents/dreamzero
python robots/droid_dreamzero_orchestrator.py \
  --robot-host "<ROBOT_LAPTOP_HOST_OR_IP>" \
  --robot-port 6000 \
  --prompt "pick up the red cup" \
  --max-chunks 30
```

Expected:

- Sends `RESET`
- Loops `RUN_CHUNK`
- Stops when server reports `done`/`truncated` or `max-chunks` reached
- Sends `CLOSE`

---

## Networking options

## Option A: Same LAN / VPN (recommended)

Use direct host/IP for all three links:

- Robot -> H100: websocket `ws://<h100>:8000`
- Robot -> PC: HTTP `http://<pc>:8001`
- Orchestrator -> Robot: TCP `<robot>:6000`

No tunnel required.

## Option B: Pinggy tunnels (if NAT/firewall blocks direct access)

Use one tunnel per externally consumed port.

### H100 (DreamZero websocket on port 8000)

```bash
ssh -p 443 -R0:localhost:8000 a.pinggy.io
```

Use the returned host/port as `--dreamzero-host` / `--dreamzero-port` on robot laptop.

### PC (Reward HTTP on port 8001)

```bash
ssh -p 443 -R0:localhost:8001 a.pinggy.io
```

Use returned host/port as `--reward-server-host` / `--reward-server-port` on robot laptop.

### Robot laptop (TCP orchestrator endpoint on port 6000)

```bash
ssh -p 443 -R0:localhost:6000 a.pinggy.io
```

Use returned host/port as `--robot-host` / `--robot-port` for orchestrator.

Notes:

- Keep tunnel terminals open.
- Verify each tunnel endpoint from consumer side before full run.
- Latency from tunnels increases MPC loop time.

---

## End-to-end quick check (before real run)

1. Start DreamZero + Reward + Robot server.
2. Run orchestrator with small candidate count first:

```bash
python robots/droid_dreamzero_client.py ... --mpc --mpc-n-candidates 2 ...
python robots/droid_dreamzero_orchestrator.py --prompt "..." --max-chunks 3
```

3. Confirm robot logs show:
   - `Planning with N=2 candidates...`
   - `Scores: [...]`
   - `Best candidate: ...`
4. Increase to `--mpc-n-candidates 8` then `16`.

---

## No-robot smoke test

If you do not have robot access right now, you can still validate the DreamZero + Reward link.

This test:

- builds a synthetic observation from local debug MP4s (or random fallback)
- calls `plan(obs, seeds)` on DreamZero websocket server
- optionally scores candidates via Reward server
- calls `commit(best_idx)` on DreamZero

### A) DreamZero-only smoke test

```bash
cd /home/adityashah/Documents/dreamzero
python eval_utils/test_mpc_pipeline_no_robot.py \
  --dreamzero-host <H100_HOST_OR_IP> \
  --dreamzero-port 8000 \
  --n-candidates 4 \
  --task "pick up the red cup"
```

### B) DreamZero + Reward smoke test

```bash
cd /home/adityashah/Documents/dreamzero
python eval_utils/test_mpc_pipeline_no_robot.py \
  --dreamzero-host <H100_HOST_OR_IP> \
  --dreamzero-port 8000 \
  --reward-host <PC_HOST_OR_IP> \
  --reward-port 8001 \
  --n-candidates 4 \
  --task "pick up the red cup"
```

If your reward endpoint is tunneled over HTTPS, add:

```bash
--reward-https
```

Success criteria:

- prints `Received N candidates`
- each candidate includes action and video shape
- prints reward scores (if reward host provided)
- prints `Commit successful. Smoke test passed.`

---

## Common failures

- `DreamZero plan failed`: Check H100 server alive, host/port reachable from robot laptop.
- `Reward scoring failed`: Check Reward server URL/port reachable from robot laptop.
- `Send RESET before RUN_CHUNK`: Orchestrator must send `RESET` first.
- Orchestrator hangs after RESET: Robot may be waiting for manual interaction or camera/robot init.
- Very slow loop: expected for serial planning with high `N`; start from `N=2`.
