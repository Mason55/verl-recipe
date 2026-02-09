# SWE Agent Recipe

将 ROCK 框架中的 SWE Agent 迁移到 VERL Recipe，验证 VERL 在 agentic RL 场景下的能力。

> **当前状态**: ✅ 简化架构 - SWE-Agent 自主管理执行环境

## 快速开始

```bash
cd /data1/lmy/agentic-rl/verl

# 1. 生成训练数据
python3.12 recipe/swe_agent/prepare/prepare_data.py --mode simple

# 2. 运行训练（唯一入口）
bash recipe/swe_agent/example/run_qwen2.5_3b.sh
```

## 执行方式

SWE-Agent 通过 subprocess 直接调用 `sweagent` 命令：
- SWE-Agent 根据自己的默认配置决定是否创建 Docker 容器
- VERL 只负责 ModelProxy 拦截和响应生成
- 不需要 VERL 参与容器生命周期管理

**优点**：
- 架构简单，易于理解和维护
- 避免 docker-in-docker 复杂性
- 与 SWE-Agent 的默认行为一致

**SWE-Agent 需求**：
```bash
# 安装 SWE-Agent（如果未安装）
cd /data1/lmy/agentic-rl/SWE-agent
python3.12 -m pip install -e .
```

## 架构

### 简化架构设计

SWE Agent Recipe 采用简洁的架构，VERL 只负责 ModelProxy 拦截，不参与容器管理：

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERL Training Loop                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SWEAgentLoop                                │   │
│  │                                                          │   │
│  │   1. Start ModelProxy (localhost:8080+)                  │   │
│  │   2. Invoke: sweagent run ...                            │   │
│  │   3. while not done:                                     │   │
│  │        request = await model_proxy.get_request()         │   │
│  │        response = await server_manager.generate()    ◄───┼── VERL 控制推理
│  │        await model_proxy.send_response(response)         │   │
│  │   4. return patch, trajectory                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ subprocess
                              ▼
              ┌───────────────────────────────┐
              │     SWE-Agent Process         │
              │  ┌─────────────────────────┐  │
              │  │  sweagent run           │  │
              │  │    ↓                    │  │
              │  │  http://127.0.0.1:8080  │──┼──→ ModelProxy
              │  │    ↓                    │  │
              │  │  Git Repo + Tools       │  │
              │  └─────────────────────────┘  │
              │ (May create Docker internally)│
              └───────────────────────────────┘
```

### 核心组件

#### 1. ModelProxy（模型拦截）
- **职责**: 拦截 SWE-Agent 的模型调用，将控制权交给 VERL
- **实现**: `model_proxy/proxy_server.py`
- **特点**:
  - 轻量级 HTTP 代理，兼容 OpenAI API 格式
  - 自动端口fallback（8080→8081→8082...）避免冲突
- **详细文档**: [Model Proxy README](model_proxy/README.md)

#### 2. Configuration Management
- **SWEAgentConfigBuilder**: 统一的配置生成器
  - 为 SWE-Agent 生成配置文件
  - 支持 unique instance ID（避免并发冲突）
  - 实现: `config/config_builder.py`

- **AgentConfigValidator**: 配置验证器
  - 验证必需字段和值的有效性
  - 提供清晰的错误信息
  - 实现: `config/config_validator.py`

#### 3. Patch Extraction
- **PatchExtractor**: 统一的补丁提取工具
  - 支持多种提取策略（.patch 文件、git diff）
  - 自动尝试多种方法
  - 实现: `utils/patch_extractor.py`

### 重构效果

通过引入这些组件，我们实现了：
- **代码复用**: 消除了配置生成和补丁提取的重复代码
- **可测试性**: 每个组件都有独立的单元测试
- **可维护性**: 职责分离，易于理解和修改
- **文档完善**: 每个组件都有清晰的文档和使用示例

详细的架构设计和设计决策，请参考 [架构文档](docs/architecture.md)。

## 目录结构

```
recipe/swe_agent/
├── config/
│   └── swe_agent_config.yaml    # Agent Loop 配置
├── example/
│   └── run_qwen2.5_3b.sh        # ✅ 唯一训练入口
├── model_proxy/
│   ├── __init__.py
│   ├── proxy_server.py          # ModelProxy 实现
│   └── README.md
├── prepare/
│   ├── prepare_data.py          # 数据生成
│   └── setup_sandbox.sh
├── reward/
│   ├── __init__.py
│   └── swe_reward.py            # 奖励函数
├── sandbox/                      # Docker 隔离
│   ├── __init__.py
│   ├── docker_sandbox.py        # DockerSandbox 实现
│   └── Dockerfile               # SWE-Agent 镜像
├── README.md
└── swe_agent_loop.py            # Agent Loop 实现
```

## 配置说明

### proxy_config

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `port` | 8080 | ModelProxy 监听端口 |
| `timeout` | 600 | 单次请求超时（秒） |

### sandbox_config

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `deployment_type` | docker | 部署模式：`docker`（隔离，生产推荐）或 `local`（快速，调试用） |
| `swe_agent_timeout` | 1800 | SWE-Agent 总执行时间（秒） |
| `max_steps` | 30 | 最大模型调用次数（`per_instance_call_limit`） |
| `max_turns` | 15 | 交互轮数上限，达到后中止当前任务 |
| `execution_timeout` | 300 | 单次工具执行超时（秒） |
| `docker_image` | swerex-python:3.11 | Docker 镜像（仅 docker 模式） |
| `docker_memory_limit` | 8g | Docker 容器内存限制（仅 docker 模式） |
| `docker_startup_timeout` | 180.0 | Docker 容器启动超时（秒，仅 docker 模式） |
| `docker_remove_container` | true | 执行后是否自动清理容器（仅 docker 模式） |

## 数据格式

训练数据需要包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | list[dict] | 对话格式的问题描述 |
| `reward_model` | dict | 奖励模型配置 |
| `extra_info` | dict | 额外信息（repo_content, expected_patch 等） |
| `agent_name` | str | Agent 名称（swe_agent） |

## 与 ROCK 的对比

| 方面 | ROCK | VERL |
|------|------|------|
| **隔离方式** | 自研 Sandbox | Docker / subprocess |
| **模型调用拦截** | ModelService + anti_call_llm | ModelProxy (HTTP) |
| **训练集成** | ROLL AgenticPipeline | VERL PPO/GRPO |
| **配置方式** | Python Config | YAML + Hydra |

## 故障排查

1. **Docker 不可用**
   - 确认 Docker Socket 已挂载
   - 确认 Docker 命令可执行

2. **ModelProxy 连接失败**
   - 检查端口是否被占用
   - 确认网络配置正确

3. **SWE-Agent 超时**
   - 增加 `swe_agent_timeout`
   - 检查任务复杂度

## 后续计划

- [ ] SWE-bench Lite 评测
- [ ] 与 ROCK 性能对比
- [ ] 分布式训练支持
