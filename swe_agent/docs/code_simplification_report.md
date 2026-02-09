# 代码简化完成报告

**日期**: 2026-02-04
**任务**: 删除 VERL Docker 管理代码，采用方案 B（让 SWE-Agent 自己管理容器）

## 完成的工作

### 1. 代码简化 (`swe_agent_loop.py`)

**删除的内容**：
- ❌ `_run_swe_agent_docker()` 方法（第 352-404 行）
- ❌ `execution_mode` 配置逻辑（第 121-128 行）
- ❌ Docker 配置相关代码
- ❌ 双模式分派逻辑

**保留并优化的内容**：
- ✅ `_run_swe_agent()` 方法（原 `_run_swe_agent_subprocess()`）
- ✅ ModelProxy 端口自动fallback机制
- ✅ Unique instance ID配置（在 `SWEAgentConfigBuilder` 中）
- ✅ PatchExtractor 统一补丁提取

**代码行数变化**：
- 之前: 592 行
- 删除: ~75 行（Docker 相关代码）
- 预计: ~515 行

### 2. 配置文件更新 (`config/swe_agent_config.yaml`)

**删除的配置**：
```yaml
# ❌ 删除
execution_mode: subprocess
docker:
  image: verl-swe-agent:latest
  memory_limit: 8g
  cpu_count: 4
  model_proxy_host: "127.0.0.1"
```

**保留的配置**：
```yaml
# ✅ 保留（简化）
sandbox_config:
  swe_agent_timeout: 1800
  max_steps: 100
  execution_timeout: 300
  output_dir: /data1/lmy/workspace/swe_agent_outputs
  python_path: /usr/bin/python3.12
```

### 3. 文档更新

**更新的文档**：
1. ✅ `README.md` - 删除双模式描述，更新架构图
2. ✅ `CLAUDE.md` - 删除 Docker 模式相关内容
3. ✅ `docs/TODO_optimizations.md` - 新建，记录方案 A 作为待优化项

**架构图变化**：
```
之前: VERL → 选择 subprocess/docker → SWE-Agent
现在: VERL → subprocess → SWE-Agent（自己决定是否用 Docker）
```

### 4. 待优化文档 (`docs/TODO_optimizations.md`)

**记录的内容**：
- 方案 A 的详细设计（像 ROCK 一样管理容器）
- 实现步骤和参考代码
- 优势分析
- 估计工作量（2-3天）

## 当前架构说明

### 执行流程

```
1. VERL 启动 ModelProxy（端口 8080+，自动递增避免冲突）
2. VERL 调用: sweagent run --config <config> --problem_statement.text <text>
3. SWE-Agent 根据自己的配置决定是否创建 Docker 容器
4. SWE-Agent 调用 http://localhost:8080/v1 (ModelProxy)
5. ModelProxy 拦截请求 → VERL 生成响应 → 返回给 SWE-Agent
6. SWE-Agent 完成后，PatchExtractor 提取补丁
7. 返回训练数据（trajectory + patch）
```

### 优势

1. **架构简单**：
   - VERL 不需要管理 Docker 容器生命周期
   - 不需要处理 docker-in-docker 复杂性
   - 代码更少，更容易维护

2. **灵活性**：
   - SWE-Agent 可以根据自己的配置选择执行环境
   - 不强制用户必须使用 Docker
   - 开发和生产环境可以使用同样的代码

3. **与 SWE-Agent 行为一致**：
   - 遵循 SWE-Agent 的默认行为
   - 减少定制化代码
   - 更容易升级 SWE-Agent 版本

### 限制

1. **容器生命周期不可控**：
   - VERL 无法提前终止 SWE-Agent 创建的容器
   - 需要依赖 SWE-Agent 的清理逻辑

2. **资源限制更难实施**：
   - 无法在 VERL 层面设置内存/CPU 限制
   - 需要在 SWE-Agent 配置中设置

3. **隔离性较弱**：
   - 多个 SWE-Agent 实例可能共享宿主机环境
   - 需要依赖 unique instance ID 避免冲突

## 待验证的问题

### 1. Unique Instance ID 是否解决并发冲突？

**之前的问题**：
```
FileExistsError: /test_repo already exists
```

**解决方案**：
```python
# config_builder.py
"name": f"verl-swe-{self.instance_id}"  # Unique per instance
```

**需要验证**：
- ✅ 端口冲突已验证解决（自动递增机制工作正常）
- ⚠️ 文件系统冲突需要进一步测试

### 2. 性能和资源使用

**需要测试**：
- 8个并发 SWE-Agent 实例的内存使用
- 是否会出现 CUDA OOM
- 训练稳定性

## 下一步建议

### 立即行动

1. **验证简化后的代码**：
   ```bash
   cd /data1/lmy/agentic-rl/verl
   bash recipe/swe_agent/example/run_qwen2.5_3b_quick.sh
   ```

2. **检查关键功能**：
   - ✅ ModelProxy 端口自动fallback
   - ⚠️ Unique instance ID 避免文件冲突
   - ⚠️ Patch 提取成功率

### 短期（1周内）

1. **性能优化**：
   - 如果遇到 CUDA OOM，调整 `gpu_memory_utilization`
   - 如果遇到文件冲突，考虑实现方案 A

2. **测试覆盖**：
   - 添加 end-to-end 测试
   - 测试多个并发实例场景

### 中期（1-2周）

1. **考虑实现方案 A**（如果需要）：
   - 参考 `docs/TODO_optimizations.md`
   - 参考 ROCK 的实现
   - 估计工作量：2-3天

## 文件变更清单

### 修改的文件

| 文件 | 变更类型 | 行数变化 |
|------|---------|---------|
| `swe_agent_loop.py` | 简化 | -75 行 |
| `config/swe_agent_config.yaml` | 简化 | -15 行 |
| `README.md` | 更新文档 | ~50 行修改 |
| `CLAUDE.md` | 更新文档 | ~30 行修改 |

### 新建的文件

| 文件 | 用途 |
|------|------|
| `docs/TODO_optimizations.md` | 记录方案 A 作为待优化项 |

### 未修改的文件

| 文件 | 原因 |
|------|------|
| `sandbox/docker_sandbox.py` | 保留以供未来实现方案 A 时参考 |
| `sandbox/Dockerfile` | 保留以供未来使用 |
| `config/config_builder.py` | 已包含 unique instance ID，无需修改 |
| `utils/patch_extractor.py` | 功能完整，无需修改 |

## 总结

✅ **完成了方案 B 的实现**：
- 删除了 VERL 的 Docker 管理代码
- 简化了架构，让 SWE-Agent 自己管理容器
- 更新了配置和文档
- 记录了方案 A 作为未来优化方向

⚠️ **需要验证**：
- Unique instance ID 是否真正解决了文件冲突
- 性能和稳定性如何

📋 **下一步**：
- 运行测试验证简化后的代码
- 根据测试结果决定是否需要实现方案 A
