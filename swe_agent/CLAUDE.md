# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **SWE Agent Recipe** for VERL (Versatile Reinforcement Learning), which implements reinforcement learning for software engineering agents. The codebase migrates SWE-Agent from the ROCK framework to VERL, enabling agentic RL training for code generation tasks.

**Core Concept**: The system intercepts SWE-Agent's model calls through a ModelProxy, allowing VERL to control the generation process and collect training trajectories. This enables RL training on software engineering tasks.

## Workspace Configuration

**IMPORTANT**: All training outputs, logs, and working files MUST be placed in `/data1/lmy/workspace/`.

This is a critical configuration requirement:
- **Training logs**: `/data1/lmy/workspace/swe_training.log` (or similar)
- **Model checkpoints**: `/data1/lmy/workspace/checkpoints/`
- **SWE-Agent outputs**: `/data1/lmy/workspace/swe_agent_outputs/`
- **Temporary files**: Use `/data1/lmy/workspace/tmp/` when needed

The configuration file (`config/swe_agent_config.yaml`) is already set to use this location:
```yaml
sandbox_config:
  output_dir: /data1/lmy/workspace/swe_agent_outputs
```

Training scripts are configured with:
```bash
default_local_dir=/data1/lmy/workspace/checkpoints/$experiment_name
```

**Do not use** `/tmp/` or other temporary directories for training outputs - always use `/data1/lmy/workspace/`.

## Key Commands

### Data Preparation
```bash
# Generate simple test data (for development)
cd /data1/lmy/agentic-rl/verl
python3.12 recipe/swe_agent/prepare/prepare_data.py --mode simple

# Generate data with custom size
python3.12 recipe/swe_agent/prepare/prepare_data.py --mode simple --train_size 100 --test_size 20
```

### Training
```bash
# Main training entry point (uses 4 GPUs by default)
bash recipe/swe_agent/example/run_qwen2.5_3b.sh

# Quick test training (reduced parameters for fast validation)
bash recipe/swe_agent/example/run_qwen2.5_3b_quick.sh
```

### Testing
```bash
# Run all unit tests
pytest tests/

# Run specific component tests
pytest tests/test_config_builder.py
pytest tests/test_config_validator.py
pytest tests/test_patch_extractor.py

# Run integration tests
pytest tests/test_integration.py

# Run with coverage
pytest --cov=config --cov=utils --cov=sandbox tests/
```

### Docker Setup (if needed by SWE-Agent)
```bash
# SWE-Agent may create its own Docker containers
# Ensure Docker is available if running inside a container
docker ps  # Should not fail
```

## Architecture Overview

### Execution Flow
1. **ModelProxy** starts an HTTP server (port 8080+) that mimics OpenAI API
2. **SWE-Agent** runs via subprocess (`sweagent run ...`)
3. **SWE-Agent** makes model calls to ModelProxy
4. **ModelProxy** suspends requests and queues them for VERL
5. **VERL** retrieves requests via `get_request()`, generates responses, and sends them back via `send_response()`
6. **SWE-Agent** receives responses and continues execution
7. **PatchExtractor** extracts the generated code patch from SWE-Agent output
8. **Reward Function** evaluates the patch quality

**Note**: SWE-Agent may create its own Docker containers internally based on its configuration. VERL does not manage these containers.

### Execution Mode

**Subprocess Mode (only mode)**:
- Invokes SWE-Agent directly using `sweagent` CLI command
- SWE-Agent decides internally whether to use Docker
- Simpler architecture, no docker-in-docker complexity

### Core Components

1. **SWEAgentLoop** (`swe_agent_loop.py`)
   - Main agent loop implementation
   - Manages ModelProxy lifecycle
   - Invokes SWE-Agent via subprocess
   - Collects trajectories for RL training

2. **ModelProxy** (`model_proxy/proxy_server.py`)
   - HTTP proxy server intercepting OpenAI API calls
   - Automatic port fallback (8080→8081→... if port occupied)
   - Implements request queuing and response synchronization

3. **Configuration System**:
   - `SWEAgentConfigBuilder`: Generates SWE-Agent YAML configs with unique instance IDs
   - `AgentConfigValidator`: Validates VERL config before execution

4. **PatchExtractor** (`utils/patch_extractor.py`)
   - Multi-strategy patch extraction (`.patch` file → `git diff HEAD` → `git diff`)
   - Unified interface for patch extraction

## Configuration Structure

Main config file: `config/swe_agent_config.yaml`

```yaml
- name: swe_agent
  _target_: recipe.swe_agent.swe_agent_loop.SWEAgentLoop

  proxy_config:
    port: 8080          # Base port (auto-increments if occupied)
    timeout: 600        # Request timeout in seconds

  sandbox_config:
    swe_agent_timeout: 1800     # Total execution timeout
    max_steps: 100              # Max model calls
    execution_timeout: 300      # Per-tool timeout
    output_dir: /data1/lmy/workspace/swe_agent_outputs
    python_path: /usr/bin/python3.12
```

## Important Implementation Details

### Port Conflict Handling
**Problem**: Multiple agent workers try to bind the same port (8080) simultaneously.

**Solution**: ModelProxy implements automatic port fallback:
- Tries port 8080
- If occupied, tries 8081, 8082, etc.
- Max 20 retries by default
- Each worker gets a unique port automatically

### Instance Isolation
Each training instance must have a unique environment name to avoid conflicts:
```python
# In config_builder.py
"name": f"verl-swe-{self.instance_id}"  # Unique per instance
```

### Import Paths
All imports use **relative paths** starting with `recipe.swe_agent`:
```python
# Correct
from recipe.swe_agent.model_proxy import ModelProxy
from recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder

# Wrong
from verl.recipe.swe_agent.model_proxy import ModelProxy  # Will fail
```

### Data Format
Training data must include:
- `prompt`: List of message dicts (conversation format)
- `reward_model`: Dict with reward config
- `extra_info`: Dict containing:
  - `repo_path`: Path to test repository
  - `expected_patch`: Expected code changes
- `agent_name`: Must be "swe_agent"

The `extra_info` field MUST be a dict, not a JSON string.

### Common Pitfalls

1. **SWE-Agent Module**: Must be installed from `/data1/lmy/agentic-rl/SWE-agent`:
   ```bash
   cd /data1/lmy/agentic-rl/SWE-agent
   python3.12 -m pip install -e .
   ```

2. **Dependency Conflicts**: If training fails with import errors, check:
   - `huggingface-hub>=0.34.0,<1.0` (not 1.x)
   - `numpy<2.0.0` (not 2.x)

3. **Test Repository**: The repo_path in training data must exist and be a valid git repo.

## Refactoring History

The codebase underwent significant refactoring to eliminate ~280 lines of duplicate code:

**Before**: Configuration generation and patch extraction logic duplicated between subprocess and Docker modes.

**After**: Extracted into reusable components:
- `SWEAgentConfigBuilder`: 216 lines (replaces ~200 lines of duplication)
- `PatchExtractor`: 142 lines (replaces ~100 lines of duplication)
- `AgentConfigValidator`: 112 lines (replaces ~50 lines embedded in main loop)

Benefits:
- Single source of truth for configs
- 100% unit test coverage for new components
- Easier to maintain and extend
- Clear separation of concerns

## Testing Strategy

- **Unit Tests**: Test individual components in isolation (Builder, Validator, Extractor)
- **Integration Tests**: Test component interactions (Loop + Builder + Validator)
- **End-to-End Tests**: Full training loop with mock data
- **Live Tests**: Real training with actual models (slow, run manually)

All tests use pytest and are located in `tests/`.

## Debugging Tips

### Check Training Status
```bash
# View training logs
tail -f /data1/lmy/workspace/swe_training.log

# Check if training process is running
ps aux | grep main_ppo

# Monitor GPU usage
nvidia-smi -l 1
```

### ModelProxy Issues
If you see "address already in use" errors, this is expected and handled automatically by port fallback. Check logs for "Model proxy server started on" to see the actual port used.

### SWE-Agent Execution Failures
Common issues:
1. **FileExistsError in _copy_repo()**: Fixed by unique instance names
2. **sweagent not found**: Install SWE-agent module
3. **Invalid config**: Use ConfigValidator to check before execution

## Output Locations

**All outputs MUST go to `/data1/lmy/workspace/`** (see Workspace Configuration section above):
- Training logs: `/data1/lmy/workspace/swe_training.log`
- Checkpoints: `/data1/lmy/workspace/checkpoints/`
- SWE-Agent outputs: `/data1/lmy/workspace/swe_agent_outputs/`

Note: Ray temp files are configured to go to `/data1/lmy/ray_tmp/` (outside workspace, for performance)

## Comparison with ROCK

| Aspect | ROCK | VERL (this codebase) |
|--------|------|---------------------|
| Model Interception | ModelService (complex) | ModelProxy HTTP (simple) |
| Environment Isolation | Custom Sandbox | Docker/subprocess |
| Configuration | Python configs | YAML + Builder pattern |
| Training Framework | ROLL AgenticPipeline | VERL PPO/GRPO |
| Patch Extraction | Scattered logic | Unified PatchExtractor |

## Python Version

Always use **python3.12** - the codebase is configured for this version:
- Training scripts specify `python3.12`
- Config explicitly sets `python_path: /usr/bin/python3.12`
- Some dependencies require 3.12+
