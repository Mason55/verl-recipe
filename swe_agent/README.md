# SWE-Agent VERL Recipe

Train language models to solve real-world software engineering tasks using reinforcement learning. This recipe integrates [SWE-agent](https://github.com/SWE-agent/SWE-agent) as the agent framework with VERL's PPO/GRPO trainer, enabling models to learn from interactive coding feedback in Docker-sandboxed environments.

## Overview

The training loop works as follows:

1. **Data**: Each training sample contains a problem statement (e.g. "fix the bug in calculator.py") and a reference patch.
2. **Rollout**: For each sample, a SWE-Agent subprocess is launched inside a Docker container. The agent interacts with a codebase by reading files, editing code, and running commands.
3. **Model Proxy**: A lightweight HTTP server intercepts the agent's LLM API calls and routes them through VERL's vLLM rollout engine, so every token the agent generates is on-policy.
4. **Reward**: After the agent finishes (or hits the turn limit), its generated patch is compared against the reference patch to produce a 0–1 reward signal.
5. **Training**: VERL applies GRPO (or PPO) policy gradient updates using the collected trajectories and rewards.

```
┌─────────────────────────────────────────────────────┐
│               VERL PPO/GRPO Trainer                 │
│  (actor, ref model, vLLM rollout, reward scoring)   │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │   SWEAgentLoop.run()    │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
     ▼                 ▼                 ▼
┌──────────┐   ┌─────────────┐   ┌──────────────┐
│ TempRepo │   │ ModelProxy  │   │ sweagent run  │
│ (git)    │   │ (HTTP)      │◄──│ (subprocess)  │
└──────────┘   └──────┬──────┘   └──────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ vLLM generate │
              │ (on-policy)   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ compute_score │
              │ (patch diff)  │
              └───────────────┘
```

## Directory Structure

```
recipe/swe_agent/
├── swe_agent_loop.py              # Core agent loop (registered as "swe_agent")
├── config/
│   ├── swe_agent_config.yaml      # Agent config (templates, tools, sandbox settings)
│   ├── runtime_config.py          # Runtime config dataclass + merge logic
│   ├── yaml_builder.py            # Generates per-instance SWE-Agent CLI YAML
│   ├── defaults.py                # Shared default templates/tool settings
│   └── __init__.py                # Config exports
├── runtime/
│   ├── model_proxy.py             # HTTP proxy: SWE-Agent ↔ vLLM
│   ├── subprocess_runner.py       # Runs `sweagent run` as subprocess
│   └── container_cleanup.py       # Docker container cleanup
├── reward/
│   └── reward.py                  # Patch-based reward function
├── prepare/
│   └── prepare_data.py            # Dataset generator (simple tasks / SWE-bench)
├── utils/
│   ├── message_utils.py           # OpenAI message normalization
│   ├── repo_manager.py            # Temp git repo creation/cleanup
│   └── patch_extractor.py         # Extract patches from .patch files or git diff
├── example/
    └── run_qwen3_4b_instruct.sh   # Single-node 8-GPU training script
```

## Prerequisites

### Hardware

- 8x NVIDIA GPUs (tested on RTX 3090 24GB, A100 also works)
- Sufficient disk space for model checkpoints (~50GB per checkpoint)

### Runtime Environment (Docker-in-Docker)

This recipe is designed to run inside a VERL Docker container. Since the training loop spawns SWE-Agent sandbox containers via Docker, the host container must be launched with **Docker-in-Docker (DinD)** support and **host networking**.

Key `docker run` flags:

| Flag | Why |
|------|-----|
| `--network host` | Required so that the ModelProxy HTTP server (inside the container) is reachable by SWE-Agent sandbox containers on the same host. Without this, sandbox containers cannot call back to the proxy. |
| `-v /var/run/docker.sock:/var/run/docker.sock` | Mounts the host Docker socket, allowing the training process to create/manage SWE-Agent sandbox containers from inside the VERL container. |
| `-v /usr/bin/docker:/usr/bin/docker:ro` | Makes the Docker CLI binary available inside the container. |

Example launch command:

```bash
# Build or pull the VERL training image first, then run:
docker run -it \
  --gpus all \
  --network host \
  --shm-size=32g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker:ro \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  --entrypoint /bin/bash \
  --name verl_swe_train \
  <your-verl-image>
```

> **Note**: You may also need to mount NVIDIA driver libraries (e.g. `libnvidia-ml.so.1`, `libcuda.so.1`) if the base image does not include them. Use `--gpus all` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)) or mount individual `.so` files as read-only volumes.

After entering the container, verify Docker access:

```bash
docker ps          # should list running containers on the host
docker pull swerex-python:3.11   # pre-pull the sandbox image
```

### Software

```bash
# 1. VERL framework (already installed in this repo)
# 2. SWE-agent CLI
pip install sweagent
which sweagent  # verify

# 3. Docker + swerex image (for sandboxed code execution)
docker pull swerex-python:3.11
docker images swerex-python:3.11  # verify

# 4. Model weights (Qwen3-4B-Instruct or your choice)
ls /path/to/models/Qwen/Qwen3-4B-Instruct-2507/config.json
```

### Verify Environment

```bash
nvidia-smi -L | wc -l                                        # expect: 8
python3 -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
#   ^ Must print real IP, NOT 127.0.x.x
docker images swerex-python:3.11 --format '{{.Repository}}'  # swerex-python
```

## Quick Start

### 1. Generate Training Data

```bash
cd /path/to/agentic-rl/verl

# Simple synthetic tasks (good for testing the pipeline)
python recipe/swe_agent/prepare/prepare_data.py \
    --mode simple \
    --train_size 8 \
    --test_size 2 \
    --output_dir data/swe_agent_test

# Or load SWE-bench Lite for real-world tasks
python recipe/swe_agent/prepare/prepare_data.py \
    --mode swebench \
    --swebench_train /path/to/swebench_train.jsonl \
    --swebench_test /path/to/swebench_test.jsonl \
    --output_dir data/swe_agent_swebench
```

The generated parquet files contain:
- `prompt`: Minimal chat-formatted problem description
- `reward_model.ground_truth`: Expected patch for reward computation
- `extra_info`: Problem statement, repo content, per-instance overrides

### 2. Launch Training (Single Node, 8 GPU)

```bash
cd /path/to/agentic-rl/verl

# Clean old checkpoints if doing a fresh run (important!)
rm -rf /path/to/workspace/checkpoints/qwen3-4b-swe-train-v1/

# Launch training
bash recipe/swe_agent/example/run_qwen3_4b_instruct.sh
```

### 3. Monitor Progress

```bash
# Training phases and expected timeline (~18 min for 2 epochs, 8 samples)
#   Ray + Model Init   ~2 min    "Loading checkpoint shards"
#   vLLM Init           ~1 min    "Capturing CUDA graphs"
#   Weights Sync        ~12s      "update_weights done"
#   Rollout (per epoch) ~2-5 min  "SWE Agent Loop completed"
#   Training Step       ~45s      "actor/pg_loss"
#   Checkpoint Save     ~2.5 min  "Saved model to"

# Ray dashboard
# http://<HEAD_IP>:8265
```

## Configuration

### Training Script Parameters

Key parameters in `example/run_qwen3_4b_instruct.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NNODES` | 1 | Number of nodes |
| `GPUS_PER_NODE` | 8 | GPUs per node |
| `train_batch_size` | 8 | Training batch size |
| `ppo_mini_batch_size` | 4 | Mini-batch size for PPO/GRPO updates |
| `max_turns` | 15 | Max agent interaction turns |
| `max_prompt_length` | 4096 | Max prompt token length |
| `max_response_length` | 16384 | Max response token length (multi-turn accumulates) |
| `infer_tp` | 8 | vLLM tensor parallel size |
| `train_sp` | 8 | Ulysses sequence parallel size |
| `gpu_memory_utilization` | 0.4 | vLLM GPU memory fraction |
| `total_epochs` | 2 | Number of training epochs |

### Agent Config (`config/swe_agent_config.yaml`)

The YAML config is the default source of truth and has two categories of fields:

**Infrastructure fields** (fixed per deployment):
- `proxy_config`: ModelProxy port, timeout, retry settings
- `sandbox_config.docker_memory_limit`: Container memory limit

**Data-affine fields** (can be overridden per instance via `extra_info`):
- `sandbox_config.max_turns`: Max interaction turns
- `sandbox_config.max_steps`: Max model call count
- `sandbox_config.docker_image`: Sandbox Docker image
- `agent.templates`: System/instance/next-step prompt templates
- `agent.tools`: Tool bundles, parse function type

Per-instance overrides are applied at runtime via `extra_info.sandbox_overrides` and `extra_info.agent_overrides`.

## Key Components

### SWEAgentLoop (`swe_agent_loop.py`)

The core agent loop, registered with VERL as `"swe_agent"`. For each episode:

1. Parses `extra_info` to get the problem statement and repo content
2. Creates a temporary git repo on disk
3. Starts a `ModelProxy` HTTP server
4. Launches `sweagent run` as a subprocess pointing at the proxy
5. Runs the interaction loop: each agent API call is intercepted, tokenized, sent to vLLM for generation, and the response is returned to the agent
6. Extracts the final patch and returns it as `AgentLoopOutput`

### ModelProxy (`runtime/model_proxy.py`)

A lightweight HTTP server that mimics the OpenAI Chat Completions API. SWE-Agent sends requests to this proxy thinking it's an LLM API. The proxy:
- Queues requests for VERL to consume
- Blocks until VERL's vLLM engine generates a response
- Returns the response to SWE-Agent

Default `proxy_config.port=0` lets the OS assign an available port for each worker (recommended).
If you set a fixed port (`port > 0`), ModelProxy falls back to auto-increment (`port+1`, `port+2`, ...)
when conflicts occur.

### Reward Function (`reward/reward.py`)

Computes a 0–1 reward by comparing the agent's generated patch against the expected patch:

| Condition | Score |
|-----------|-------|
| Exact patch match | 1.0 |
| All changed files match | 0.5 |
| Partial file overlap | 0.2 + 0.3 × overlap_ratio |
| Patch generated but no overlap | 0.1 |
| No patch generated | 0.0 |

### Data Preparation (`prepare/prepare_data.py`)

Two modes:
- **`simple`**: Generates small synthetic tasks (rename file, create file, fix bug) for pipeline testing
- **`swebench`**: Loads SWE-bench Lite instances with real GitHub issues and patches

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Training exits at 100% immediately | Old checkpoint matches `total_epochs` | `rm -rf $WORK_BASE/checkpoints/$EXPERIMENT_NAME/` |
| Fixed proxy port already in use (`port > 0`) | ModelProxy falls back to next port (`port+1`, ...) | Keep `port: 0` (recommended) or increase `max_port_retries` |
| SWE-Agent TimeoutError | Docker container startup timeout | Pre-pull: `docker pull swerex-python:3.11` |
| OOM during rollout | Too many concurrent Docker containers | Reduce `train_batch_size` or `docker_memory_limit` |
| No patch found | Agent didn't run `submit` | Increase `max_turns` or improve system prompt |

### Emergency Cleanup

```bash
# Stop all SWE-Agent Docker containers
docker ps --filter "ancestor=swerex-python:3.11" -q | xargs -r docker stop

# Force stop Ray
ray stop --force

# Kill training process
pkill -9 -f main_ppo
```

## Extending

### Custom Tasks

Create your own training data by adding new task generators in `prepare/prepare_data.py`. Each task needs:
- `problem_statement`: Natural language description
- `repo_content`: Dict mapping file paths to content (the starting codebase)
- `expected_patch`: The correct unified diff

### Custom Reward Functions

Replace or extend `reward/reward.py`. The function signature is:

```python
def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    """Returns a float reward in [0, 1]."""
```

### Custom Templates

Override prompt templates per-instance via `extra_info.agent_overrides.templates` or globally in `swe_agent_config.yaml`.
