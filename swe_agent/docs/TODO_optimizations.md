# SWE Agent Recipe - 待优化项

## 优先级 P0 - 高优先级优化

### 1. 像 ROCK 一样管理 Docker 容器（方案 A）

**当前状态（方案 B）**：
- VERL 直接调用 `sweagent` 命令（subprocess 模式）
- SWE-Agent 根据自己的默认配置决定是否创建 Docker 容器
- VERL 不参与容器生命周期管理

**目标架构（方案 A - 参考 ROCK）**：
```
┌─────────────────────────────────────────┐
│     VERL Process                        │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Docker Container (VERL管理)   │   │
│  │                                 │   │
│  │  ┌─────────────────────────────┐│   │
│  │  │   SWE-Agent                 ││   │
│  │  │   (preexisting mode)        ││   │
│  │  └─────────────────────────────┘│   │
│  │                                 │   │
│  └─────────────────────────────────┘   │
│         ↑                               │
│  ┌──────────────┐                      │
│  │ ModelProxy   │                      │
│  │ (localhost)  │                      │
│  └──────────────┘                      │
└─────────────────────────────────────────┘
```

**优化内容**：

1. **恢复 DockerSandbox，但修改实现**：
   - 创建 Docker 容器（使用 `--privileged` 或 `--network host`）
   - 挂载代码仓库到容器内
   - 在容器内生成 SWE-Agent 配置文件
   - **关键修改**：配置 SWE-Agent 使用 `preexisting` 或 `local` 部署模式，而不是 `docker` 模式

2. **SWE-Agent 配置修改**：
   ```yaml
   env:
     deployment:
       type: local  # 或 preexisting，不是 docker
     repo:
       type: preexisting  # 使用容器内已有的代码
       path: /workspace/repo
   ```

3. **容器启动命令**：
   ```bash
   docker run \
     --privileged \
     --network host \
     -v /workspace:/workspace \
     -e OPENAI_API_BASE=http://localhost:8080/v1 \
     verl-swe-agent:latest \
     sweagent run --config /workspace/config.yaml
   ```

**优势**：
- ✅ 完全隔离的执行环境
- ✅ 容器生命周期可控（可以提前终止、清理）
- ✅ 避免多个 SWE-Agent 实例之间的文件系统冲突
- ✅ 与 ROCK 架构对齐，经过生产验证
- ✅ 可以设置资源限制（memory_limit, cpu_count）

**实现参考**：
- ROCK 代码：`/data1/lmy/agentic-rl/ROCK/rock/deployments/docker.py`
- ROCK SWE-Agent：`/data1/lmy/agentic-rl/ROCK/rock/sdk/sandbox/agent/swe_agent.py`
- ROCK 测试：`/data1/lmy/agentic-rl/ROCK/tests/integration/sdk/sandbox/agent/swe_agent/`

**估计工作量**：2-3 天
- 修改 `DockerSandbox` 实现（1天）
- 修改 `SWEAgentConfigBuilder` 支持 preexisting 模式（0.5天）
- 测试和调试（1-1.5天）

---

## 优先级 P1 - 中优先级优化

### 2. 优化 ModelProxy 端口分配策略

**当前实现**：
- 自动递增端口（8080 → 8081 → 8082 ...）
- 最多重试 20 次

**优化方案**：
- 使用操作系统分配的随机可用端口（port=0）
- 记录实际绑定的端口，传递给 SWE-Agent

**优势**：
- 完全避免端口冲突
- 不需要重试逻辑

### 3. 支持分布式训练场景

**当前限制**：
- ModelProxy 绑定 localhost
- 只能在单机上运行

**优化方案**：
- 支持 ModelProxy 绑定到 0.0.0.0
- SWE-Agent 通过网络地址访问 ModelProxy
- 需要考虑安全性（认证、加密）

### 4. 性能监控和指标收集

**需要监控的指标**：
- SWE-Agent 执行时间
- ModelProxy 请求/响应延迟
- Patch 提取成功率
- 容器创建和销毁时间（如果实现方案 A）

**实现方式**：
- 集成 Prometheus metrics
- 输出结构化日志
- 集成到 VERL 的 metrics 系统

---

## 优先级 P2 - 低优先级优化

### 5. 支持更多 SWE-Agent 配置选项

**当前支持**：
- 基本的 agent templates
- 基本的 tool bundles

**可以扩展**：
- 自定义 agent prompts
- 更多 tool bundles
- 环境变量配置

### 6. 改进错误处理和重试逻辑

**当前状态**：
- SWE-Agent 失败直接返回 None
- 没有自动重试

**优化方案**：
- 区分不同类型的错误（超时、崩溃、配置错误）
- 对某些错误进行自动重试
- 提供更详细的错误信息

### 7. 优化 Patch 提取策略

**当前策略**：
1. 尝试读取 .patch 文件
2. 尝试 `git diff HEAD`
3. 尝试 `git diff`

**优化方向**：
- 支持更多 patch 格式
- 更智能的 patch 验证
- 支持增量 patch（只提取关键更改）

---

## 技术债务

### 1. 测试覆盖率

**当前状态**：
- 核心组件有单元测试（ConfigBuilder, PatchExtractor, ConfigValidator）
- 缺少 SWEAgentLoop 的集成测试

**需要补充**：
- End-to-end 测试
- 错误场景测试
- 性能测试

### 2. 文档完善

**需要补充**：
- 故障排查指南
- 性能调优指南
- 与 ROCK 的详细对比

### 3. 代码清理

**需要清理的部分**：
- ~~DockerSandbox 相关代码~~（已在方案 B 中删除）
- 不再使用的配置选项
- 临时文件清理逻辑

---

## 参考资料

- **ROCK 源码**：`/data1/lmy/agentic-rl/ROCK/`
- **SWE-Agent 文档**：https://github.com/princeton-nlp/SWE-agent
- **VERL 文档**：内部 Wiki

---

## 变更记录

| 日期 | 变更内容 | 负责人 |
|------|---------|--------|
| 2026-02-04 | 创建文档，记录方案 A 作为待优化项 | Claude |
| | 当前采用方案 B（让 SWE-Agent 自己管理容器） | |
