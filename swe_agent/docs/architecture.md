# SWE Agent 架构文档

本文档详细介绍 SWE Agent Recipe 的架构设计、重构过程以及设计决策。

## 设计理念

### 与 ROCK 模式对齐

SWE Agent Recipe 从 ROCK 框架迁移而来，保持了其核心设计理念的同时，进行了简化和优化：

| 设计原则 | ROCK 实现 | VERL 实现 | 改进 |
|---------|----------|-----------|------|
| **模型拦截** | ModelService + anti_call_llm | ModelProxy (HTTP) | 更轻量，更易理解 |
| **环境隔离** | 自研 Sandbox | Docker + subprocess | 标准化，易维护 |
| **配置管理** | Python Config | YAML + Builder Pattern | 可复用，易测试 |
| **补丁提取** | 分散在多处 | PatchExtractor 统一 | 单一职责，易扩展 |

### 核心设计目标

1. **简洁性**: 每个组件只做一件事，并把它做好
2. **可测试性**: 所有组件都有独立的单元测试
3. **可复用性**: 通过 Builder 和 Extractor 模式消除重复
4. **可维护性**: 清晰的职责分离，易于理解和修改

## 组件交互

### 高层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VERL Training Loop                          │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    SWEAgentLoop                             │   │
│  │                                                             │   │
│  │  ┌──────────────┐    ┌─────────────┐    ┌───────────────┐ │   │
│  │  │ Config       │    │ Model       │    │ Patch         │ │   │
│  │  │ Validator    │───▶│ Proxy       │───▶│ Extractor     │ │   │
│  │  └──────────────┘    └─────────────┘    └───────────────┘ │   │
│  │         │                    │                    │         │   │
│  │         ▼                    ▼                    ▼         │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │         Execution: Subprocess or Docker              │  │   │
│  │  │                                                      │  │   │
│  │  │  ┌────────────────┐                                │  │   │
│  │  │  │ Config Builder │ ──▶ SWE-Agent YAML Config     │  │   │
│  │  │  └────────────────┘                                │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  ServerManager (VERL)                       │   │
│  │  • 接收 ModelProxy 转发的请求                                │   │
│  │  • 生成响应并返回                                            │   │
│  │  • 控制整个推理过程                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 组件依赖关系

```
SWEAgentLoop
    ├── AgentConfigValidator (验证配置)
    │   └── 验证 proxy_config 和 sandbox_config
    │
    ├── ModelProxy (模型拦截)
    │   ├── 启动 HTTP 服务器
    │   ├── 接收 SWE-Agent 的模型请求
    │   ├── 转发给 VERL ServerManager
    │   └── 返回响应给 SWE-Agent
    │
    ├── SWEAgentConfigBuilder (配置生成)
    │   ├── 生成 SWE-Agent YAML 配置
    │   ├── 配置 ModelProxy 端点
    │   └── 设置执行参数（timeout, max_steps）
    │
    ├── Subprocess/Docker Execution (执行环境)
    │   ├── Subprocess: 直接运行 SWE-Agent
    │   └── DockerSandbox: 容器化运行
    │       ├── 构建隔离环境
    │       ├── 挂载代码和配置
    │       ├── 管理容器生命周期
    │       └── 清理资源
    │
    └── PatchExtractor (补丁提取)
        ├── 策略 1: 读取 .patch 文件
        ├── 策略 2: git diff HEAD
        └── 策略 3: git diff
```

## 代码指标

### 重构前后对比

| 组件 | 重构前 | 重构后 | 改进 |
|-----|-------|-------|------|
| **配置生成代码** | 分散在 swe_agent_loop.py 和 docker_sandbox.py 中 (~200 行重复) | SWEAgentConfigBuilder (216 行) | ✅ 统一管理，消除重复 |
| **补丁提取代码** | 分散在两个文件中 (~100 行重复) | PatchExtractor (142 行) | ✅ 统一接口，多策略支持 |
| **配置验证** | 嵌入在主循环中 (~50 行) | AgentConfigValidator (112 行) | ✅ 独立验证，更完善 |
| **总代码行数** | ~1800 行 | ~1520 行 | ✅ 减少 ~15% |
| **单元测试覆盖** | 无 | 100% (新组件) | ✅ 测试完善 |

### 文件结构

```
recipe/swe_agent/
├── config/
│   ├── __init__.py                   # 导出 SWEAgentConfigBuilder, AgentConfigValidator
│   ├── config_builder.py             # 216 行 - 配置生成器
│   ├── config_validator.py           # 112 行 - 配置验证器
│   └── swe_agent_config.yaml         # VERL 配置文件
│
├── utils/
│   ├── __init__.py                   # 导出 PatchExtractor
│   └── patch_extractor.py            # 142 行 - 补丁提取器
│
├── model_proxy/
│   ├── __init__.py                   # 导出 ModelProxy, ModelRequest
│   ├── proxy_server.py               # 模型代理服务器
│   └── README.md                     # 详细文档
│
├── sandbox/
│   ├── __init__.py                   # 导出 DockerSandbox
│   ├── docker_sandbox.py             # 462 行 - Docker 执行环境
│   └── Dockerfile                    # SWE-Agent 镜像
│
├── tests/
│   ├── test_config_builder.py        # ConfigBuilder 单元测试
│   ├── test_config_validator.py      # ConfigValidator 单元测试
│   ├── test_patch_extractor.py       # PatchExtractor 单元测试
│   └── test_integration.py           # 集成测试
│
├── docs/
│   ├── architecture.md               # 本文档
│   └── plans/                        # 重构计划
│
├── swe_agent_loop.py                 # 588 行 - 主循环
└── README.md                         # 使用文档
```

## 测试策略

### 单元测试

每个新组件都有完整的单元测试：

#### 1. SWEAgentConfigBuilder Tests (`tests/test_config_builder.py`)

测试范围：
- ✅ 基本配置生成
- ✅ YAML 序列化
- ✅ 文件写入
- ✅ 自定义模板
- ✅ 自定义环境变量
- ✅ 自定义工具包

#### 2. AgentConfigValidator Tests (`tests/test_config_validator.py`)

测试范围：
- ✅ 缺失配置检测
- ✅ proxy_config 验证（端口号）
- ✅ sandbox_config 验证（timeout, max_steps）
- ✅ 类型检查
- ✅ 错误消息（中文）

#### 3. PatchExtractor Tests (`tests/test_patch_extractor.py`)

测试范围：
- ✅ .patch 文件提取
- ✅ git diff HEAD 提取
- ✅ git diff 提取
- ✅ 多策略回退
- ✅ 错误处理

### 集成测试

集成测试验证组件间的协作（`tests/test_integration.py`）：

- ✅ SWEAgentLoop + ConfigBuilder + ConfigValidator
- ✅ SWEAgentLoop + PatchExtractor
- ✅ DockerSandbox + ConfigBuilder + PatchExtractor
- ✅ 端到端工作流（配置 → 执行 → 提取补丁）

### 测试命令

```bash
# 运行所有测试
pytest tests/

# 运行单个组件测试
pytest tests/test_config_builder.py
pytest tests/test_config_validator.py
pytest tests/test_patch_extractor.py

# 运行集成测试
pytest tests/test_integration.py

# 查看覆盖率
pytest --cov=config --cov=utils --cov=sandbox tests/
```

## 设计模式

### 1. Builder Pattern (构建器模式)

**应用**: `SWEAgentConfigBuilder`

**优点**:
- 分离配置构建逻辑和使用逻辑
- 支持逐步构建复杂配置
- 易于测试和扩展

**示例**:
```python
builder = SWEAgentConfigBuilder(
    instance_id="test-123",
    repo_path="/workspace/repo",
    output_dir="/tmp/output",
    model_proxy_port=8080,
)
config = builder.build()
yaml_str = builder.to_yaml()
builder.to_file("/tmp/config.yaml")
```

### 2. Strategy Pattern (策略模式)

**应用**: `PatchExtractor`

**优点**:
- 支持多种提取策略
- 自动回退机制
- 易于添加新策略

**示例**:
```python
extractor = PatchExtractor(
    output_dir="/tmp/output",
    instance_id="test-123",
    repo_path="/workspace/repo",
)
# 自动尝试: .patch 文件 → git diff HEAD → git diff
patch = await extractor.extract()
```

### 3. Validator Pattern (验证器模式)

**应用**: `AgentConfigValidator`

**优点**:
- 集中验证逻辑
- 清晰的错误消息
- 易于测试和维护

**示例**:
```python
validator = AgentConfigValidator(config)
try:
    validator.validate()
except ConfigValidationError as e:
    print(f"配置错误: {e}")
```

## 执行流程

### Subprocess 模式

```
1. SWEAgentLoop 启动
   ├── AgentConfigValidator.validate()
   └── ModelProxy.start_server()

2. 生成配置
   ├── SWEAgentConfigBuilder.build()
   └── 写入临时 YAML 文件

3. 启动 SWE-Agent (subprocess)
   └── python -m swe_agent.run --config /tmp/config.yaml

4. 交互循环
   ├── ModelProxy.get_request()
   ├── ServerManager.generate()
   └── ModelProxy.send_response()

5. 提取补丁
   ├── PatchExtractor.extract()
   └── 返回 (patch, trajectory)

6. 清理
   ├── 等待进程结束
   ├── ModelProxy.stop_server()
   └── 删除临时文件
```

### Docker 模式

```
1. SWEAgentLoop 启动
   ├── AgentConfigValidator.validate()
   └── ModelProxy.start_server()

2. DockerSandbox 准备
   ├── 创建临时工作目录
   ├── 克隆代码仓库
   ├── SWEAgentConfigBuilder.build()
   └── 写入配置文件

3. 启动 Docker 容器
   ├── docker run --network host
   ├── 挂载代码和配置
   └── 执行 SWE-Agent

4. 交互循环
   ├── ModelProxy.get_request()
   ├── ServerManager.generate()
   └── ModelProxy.send_response()

5. 提取补丁
   ├── PatchExtractor.extract()
   └── 返回 (patch, trajectory)

6. 清理
   ├── docker stop/rm
   ├── ModelProxy.stop_server()
   └── 删除临时目录
```

## 配置管理

### 配置层次

```
VERL 配置 (swe_agent_config.yaml)
    └── agent
        ├── proxy_config
        │   ├── port: 8080
        │   └── timeout: 600
        │
        └── sandbox_config
            ├── execution_mode: subprocess/docker
            ├── swe_agent_timeout: 1800
            ├── max_steps: 100
            ├── execution_timeout: 300
            └── docker (仅 Docker 模式)
                ├── image: verl-swe-agent:latest
                ├── memory_limit: 8g
                └── cpu_count: 4
```

### 配置流程

```
1. Hydra 加载 swe_agent_config.yaml
   └── 生成 config dict

2. AgentConfigValidator 验证
   ├── 检查必需字段
   ├── 验证值的有效性
   └── 返回清晰的错误消息

3. SWEAgentConfigBuilder 构建
   ├── 读取 proxy_config 和 sandbox_config
   ├── 生成 SWE-Agent YAML 配置
   └── 写入临时文件

4. SWE-Agent 使用配置
   └── 连接到 ModelProxy 端点
```

## 错误处理

### 配置错误

```python
try:
    validator = AgentConfigValidator(config)
    validator.validate()
except ConfigValidationError as e:
    # 清晰的中文错误消息
    print(f"配置错误: {e}")
    # 例如: "proxy_config.port 必须是有效的端口号 (1-65535)，当前值: 0"
```

### 执行错误

```python
try:
    patch, trajectory = await agent_loop(...)
except asyncio.TimeoutError:
    # SWE-Agent 超时
    logger.error("SWE-Agent 执行超时")
except Exception as e:
    # 其他错误
    logger.error(f"执行失败: {e}")
```

### 补丁提取错误

```python
extractor = PatchExtractor(...)
patch = await extractor.extract()
if patch is None:
    # 所有策略都失败
    logger.warning("未能提取补丁")
```

## 最佳实践

### 1. 配置管理

- ✅ 使用 `SWEAgentConfigBuilder` 而不是手动构建 YAML
- ✅ 始终使用 `AgentConfigValidator` 验证配置
- ✅ 将自定义模板和环境变量传递给 Builder

### 2. 补丁提取

- ✅ 使用 `PatchExtractor` 统一提取补丁
- ✅ 提供 `repo_path` 启用 git diff 回退
- ✅ 检查返回值是否为 `None`

### 3. 测试

- ✅ 为新组件编写单元测试
- ✅ 使用集成测试验证工作流
- ✅ 测试错误情况和边界条件

### 4. Docker 模式

- ✅ 确保 Docker Socket 已挂载
- ✅ 使用 `--network host` 访问 ModelProxy
- ✅ 设置合理的资源限制

## 与 ROCK 对比

### 相同点

- ✅ 模型拦截机制（anti-call）
- ✅ 隔离执行环境
- ✅ 补丁提取和验证
- ✅ 轨迹收集

### 不同点

| 方面 | ROCK | VERL | 优势 |
|-----|------|------|-----|
| **模型拦截** | ModelService (复杂) | ModelProxy (HTTP) | 更简单，标准化 |
| **配置管理** | Python Config | SWEAgentConfigBuilder | 可测试，可复用 |
| **补丁提取** | 分散逻辑 | PatchExtractor | 统一接口，多策略 |
| **配置验证** | 嵌入主循环 | AgentConfigValidator | 独立验证，清晰错误 |
| **隔离方式** | 自研 Sandbox | Docker | 标准化，易维护 |
| **测试** | 部分覆盖 | 完整单元测试 | 更高质量 |

## 后续改进

### 短期

- [ ] 添加更多单元测试
- [ ] 优化 Docker 镜像大小
- [ ] 支持更多 SWE-Agent 工具包
- [ ] 添加性能监控

### 中期

- [ ] 支持分布式训练
- [ ] SWE-bench Lite 评测
- [ ] 与 ROCK 性能对比
- [ ] 优化补丁验证逻辑

### 长期

- [ ] 支持更多 Agent 框架
- [ ] 通用化 AgentLoop 抽象
- [ ] 云端部署支持
- [ ] 自动化调优

## 总结

通过引入 `SWEAgentConfigBuilder`、`PatchExtractor` 和 `AgentConfigValidator` 三个核心组件，我们实现了：

1. **代码复用**: 消除了 ~280 行重复代码
2. **可测试性**: 100% 单元测试覆盖（新组件）
3. **可维护性**: 清晰的职责分离
4. **可扩展性**: 易于添加新策略和配置
5. **文档完善**: 每个组件都有清晰的文档

这些改进使 SWE Agent Recipe 更加简洁、可靠和易于维护，为后续的功能扩展和性能优化奠定了坚实基础。
