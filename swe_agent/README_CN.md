# SWE-Agent + VERL：用强化学习训练 代码修复智能体

## 1. 项目背景：

### 1.1 大模型写代码的现状

当前的大语言模型（如Qwen、DeepSeek 等）已经具备了不错的代码生成能力，但在面对**真实的软件工程任务**时仍然存在明显短板：

- **单轮生成 vs 多步交互**：真实的 bug 修复需要"阅读代码 → 定位问题 → 修改代码 → 验证结果"这样的多步推理，而不是一次性输出完整答案。
- **缺乏环境反馈**：模型在训练阶段看到的是静态的 (prompt, answer) 对，从未体验过"执行命令后看到报错，再调整策略"这种交互式学习。
- **SFT 的天花板**：Supervised Fine-Tuning（监督微调）只能教会模型模仿专家轨迹，无法让模型从自己的错误中学习——而后者正是 RL（强化学习）的核心优势。

### 1.2 SWE-Agent 是什么？

[SWE-Agent](https://github.com/SWE-agent/SWE-agent) 是由普林斯顿大学开发的开源软件工程智能体框架。它的核心理念是：

> 让 LLM 像一个真正的开发者一样在终端中工作——浏览文件、编辑代码、执行命令、提交补丁。

SWE-Agent 提供了一整套工具链：
- **文件浏览**：`cat`、`ls`、`find` 等标准 bash 命令
- **代码编辑**：`str_replace_editor`（Anthropic 风格的精确文本替换工具）
- **代码审查**：`submit` 时自动 review 修改内容
- **沙箱执行**：所有操作都在 Docker 容器中进行，安全隔离

SWE-Agent 在 [SWE-bench](https://www.swebench.com/) 基准测试上取得了很好的成绩，但它**依赖闭源的强模型**（如 Claude、GPT-4）作为大脑。我们的目标是：**用强化学习训练开源模型，使其也能胜任这些任务。**

### 1.3 我们的目标

将 SWE-Agent 的交互式代码修复框架与 VERL 的分布式强化学习训练框架结合，实现：

1. **端到端 RL 训练**：模型在 Docker 沙箱中真实地执行代码修复任务，通过 patch 对比获得奖励信号，进行策略梯度更新。
2. **On-policy 生成**：每一个 token 都由当前策略模型生成（而非回放旧轨迹），保证训练信号的新鲜度。
3. **多轮交互学习**：模型学会"探索 → 理解 → 修改 → 验证 → 提交"的完整工作流，而不是死记硬背答案。

---

## 2. 系统架构：如何在 VERL 中集成 SWE-Agent？

### 2.1 整体工作流

```
┌──────────────────────────────────────────────────────┐
│                VERL PPO/GRPO Trainer                  │
│   (actor, ref model, vLLM rollout, reward scoring)   │
└───────────────────────┬──────────────────────────────┘
                        │  每个 episode
           ┌────────────┴────────────┐
           │   SWEAgentLoop.run()    │
           └────────────┬────────────┘
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
┌──────────┐   ┌──────────────┐   ┌───────────────┐
│ TempRepo │   │  ModelProxy   │   │ sweagent run   │
│  (git)   │   │  (HTTP 代理)  │◄──│  (子进程)       │
└──────────┘   └──────┬───────┘   └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ vLLM generate │
              │  (on-policy)  │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ compute_score │
              │ (patch 对比)   │
              └───────────────┘
```

训练循环的每一步：

1. **数据加载**：从 parquet 文件读取训练样本，每个样本包含问题描述和参考补丁（ground truth patch）。
2. **环境准备**：为每个样本创建临时 git 仓库，将源代码写入其中。
3. **启动代理**：启动 ModelProxy HTTP 服务器 + SWE-Agent 子进程。SWE-Agent 以为自己在调用 OpenAI API，实际上请求被 ModelProxy 截获。
4. **多轮交互**：ModelProxy 将 SWE-Agent 的请求传给 VERL 的 vLLM 引擎生成回复，回复再返回给 SWE-Agent 执行。如此循环直到 Agent 提交补丁或达到轮次上限。
5. **奖励计算**：将 Agent 生成的 patch 与参考 patch 对比，计算 0~1 的奖励分数。
6. **策略更新**：VERL 使用 GRPO（或 PPO）算法，基于收集到的多轮轨迹和奖励信号进行策略梯度更新。

### 2.2 关键设计：ModelProxy 反向调用机制

这是整个集成方案中最巧妙的部分。传统的 Agent 框架（如 SWE-Agent）都假设自己在调用一个远程 LLM API。但在 RL 训练中，我们需要：

- 每个 token 都由**当前策略模型**生成（on-policy）
- 记录每个 token 的 **log probability**（用于策略梯度计算）
- 区分**模型生成的 token**（需要计算梯度）和**环境反馈的 token**（观察，不需要梯度）

ModelProxy 的解决方案：

```
SWE-Agent                    ModelProxy                      VERL / vLLM
   │                             │                               │
   │── POST /v1/chat/completions ─>│                               │
   │                             │── 挂起请求，放入队列 ──────>│
   │                             │                               │── get_request()
   │                             │                               │── tokenize messages
   │                             │                               │── vLLM generate
   │                             │                               │── 记录 token ids + logprobs
   │                             │<── send_response(text) ────│
   │<── OpenAI 格式响应 ─────────│                               │
   │                             │                               │
   │  (执行命令，获得观察)         │                               │
   │── POST /v1/chat/completions ─>│    ... 下一轮 ...            │
```

这种"反向调用"机制让我们可以**完全不修改 SWE-Agent 的源码**，就能将其集成到 VERL 的训练循环中。

### 2.3 Token 级别的轨迹收集

在多轮交互中，完整的 token 序列由两部分交替组成：

| 类型 | 来源 | mask | 说明 |
|------|------|------|------|
| 模型回复 | vLLM 生成 | `1` | 需要计算策略梯度 |
| 环境观察 | SWE-Agent 工具输出 | `0` | 不参与梯度计算，仅作为上下文 |

例如一个 3 轮交互的 token 序列：

```
[模型回复1 (mask=1)] [观察1 (mask=0)] [模型回复2 (mask=1)] [观察2 (mask=0)] [模型回复3 (mask=1)]
```

这样 VERL 的 PPO/GRPO 更新只会对模型自己生成的 token 施加梯度，而环境观察部分作为条件上下文存在。

### 2.4 奖励函数设计

奖励函数 (`reward/reward.py`) 采用渐进式评分策略：

| 条件 | 得分 | 含义 |
|------|------|------|
| Patch 完全匹配 | **1.0** | Agent 生成了正确的修复补丁 |
| 修改了所有正确的文件 | **0.5** | 方向对了，但具体修改有差异 |
| 部分文件重叠 | **0.2 + 0.3 × overlap** | 部分正确，给予部分奖励 |
| 生成了 patch 但文件不匹配 | **0.1** | 至少尝试了修改（鼓励探索） |
| 没有生成 patch | **0.0** | Agent 未能完成任务 |

这种阶梯式奖励比简单的 0/1 二值奖励更有效——它为模型提供了更丰富的梯度信号，让模型能逐步学会"先找对文件，再写对补丁"。

---

## 3. 实际运行效果

### 3.1 训练配置

我们使用 **Qwen3-4B-Instruct** 作为基座模型，在 8×GPU 节点上进行训练。关键参数：

| 参数 | 值 | 说明 |
|------|------|------|
| 基座模型 | Qwen3-4B-Instruct | 40亿参数的指令微调模型 |
| 算法 | GRPO | Group Relative Policy Optimization |
| 最大交互轮次 | 5~15 | 简单任务 5 轮，复杂任务 15 轮 |
| 训练 batch size | 8 | 每步 8 个并发 episode |
| 张量并行 | 8-way TP | 8 GPU 并行推理 |
| vLLM 显存占比 | 0.5 | 给 KV cache 留出充足空间 |

### 3.2 训练数据类型

系统支持两类训练数据：

**1. 简单合成任务（Simple）**——用于流水线验证：
- 文件重命名、创建新文件、修复 bug、添加 docstring 等
- 每个任务只需 2~4 轮交互
- 共 20 种不同任务模板，循环采样

**2. SWE-bench 真实任务**——用于正式训练：
- 来自 GitHub 真实 issue 的代码修复任务
- 每个任务有对应的 Docker 镜像、测试用例、gold patch
- 通常需要 10~30 轮交互

---

## 4. 交互示例：一个完整的 Agent 轨迹

下面以一个简单但完整的任务为例，展示 SWE-Agent 在一次 episode 中的多轮交互过程。

### 4.1 任务描述

> **问题**："Fix the bug in calculator.py: the add function should return a + b, not a - b"
>
> **仓库内容**：
> ```python
> # calculator.py
> def add(a, b):
>     return a - b
> ```
>
> **期望补丁**：将 `return a - b` 修改为 `return a + b`

### 4.2 多轮交互轨迹

下面是 Agent 与环境交互的完整过程（为便于理解，已简化格式）：

---

**系统提示（由 SWE-Agent 注入）：**

```
You are a helpful assistant that can interact with a computer to solve tasks.

IMPORTANT: Every response MUST follow this exact format:

DISCUSSION
Your reasoning about what to do next.

```
exactly_one_command_here
```

Rules:
- Include EXACTLY ONE code block per response
- When you are done, run the `submit` command to submit your changes
```

---

**第 1 轮：Agent 接收问题并探索仓库**

*实例提示（Instance Template）：*
```
I've uploaded a python code repository in the directory /tmp/repo_xxx.

Fix the bug in calculator.py: the add function should return a + b, not a - b.

Implement the necessary changes. You MUST run `submit` when you are done.
```

*模型回复（vLLM 生成，mask=1）：*
```
DISCUSSION
Let me first look at the current content of calculator.py to understand the bug.

```

```
```

*SWE-Agent 执行命令，返回观察（环境输出，mask=0）：*
```
OBSERVATION:
def add(a, b):
    return a - b
```

---

**第 2 轮：Agent 修改代码**

*模型回复（vLLM 生成，mask=1）：*
```
DISCUSSION
I can see the bug. The add function returns `a - b` instead of `a + b`.
I'll fix this using the str_replace_editor tool.

```
str_replace_editor /tmp/repo_xxx/calculator.py <<'EOF'
old_str:     return a - b
new_str:     return a + b
EOF
```
```

*SWE-Agent 执行编辑，返回观察（环境输出，mask=0）：*
```
OBSERVATION:
The file /tmp/repo_xxx/calculator.py has been edited.
```

---

**第 3 轮：Agent 验证修改并提交**

*模型回复（vLLM 生成，mask=1）：*
```
DISCUSSION
The fix has been applied. Let me verify the change looks correct and then submit.

```
cat calculator.py
```
```

*SWE-Agent 执行命令，返回观察（环境输出，mask=0）：*
```
OBSERVATION:
def add(a, b):
    return a + b
```

---

**第 4 轮：Agent 确认无误，提交补丁**

*模型回复（vLLM 生成，mask=1）：*
```
DISCUSSION
The fix looks correct. The add function now returns a + b. Let me submit the changes.

```
submit
```
```

*SWE-Agent 执行 submit，生成 patch 并退出。*

---

### 4.3 轨迹中的 Token 流

整个交互产生的 token 序列如下所示：

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [初始 prompt ids]                                                       │
│ (系统提示 + 实例描述，由 chat template 编码)                              │
├─────────────────────────────────────────────────────────────────────────┤
│ [回复 1: "DISCUSSION...cat calculator.py"]          mask=1, logprobs=✓  │
│ [观察 1: "def add(a, b):\n    return a - b"]        mask=0, logprobs=0  │
│ [回复 2: "DISCUSSION...str_replace_editor..."]      mask=1, logprobs=✓  │
│ [观察 2: "The file has been edited."]               mask=0, logprobs=0  │
│ [回复 3: "DISCUSSION...cat calculator.py"]          mask=1, logprobs=✓  │
│ [观察 3: "def add(a, b):\n    return a + b"]        mask=0, logprobs=0  │
│ [回复 4: "DISCUSSION...submit"]                     mask=1, logprobs=✓  │
└─────────────────────────────────────────────────────────────────────────┘
```

VERL 的策略梯度更新仅作用于 `mask=1` 的部分（模型自己生成的回复），`mask=0` 的环境观察作为条件上下文，引导模型学会根据执行结果调整策略。

### 4.4 奖励计算

Agent 提交后，系统提取生成的 patch：

```diff
diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
```

与参考 patch 进行归一化后对比：**完全匹配** → 奖励 = **1.0**

这个奖励信号通过 GRPO 算法反传，强化模型在类似场景下"先读文件 → 精确修改 → 验证 → 提交"的行为模式。

---

## 5. 更复杂的 SWE-bench 真实任务示例

上面的简单任务只需 4 轮。在 SWE-bench 的真实 GitHub issue 中，交互通常更加复杂：

```
轮次 1:  ls -la                                    # 了解项目结构
轮次 2:  find . -name "*.py" | head -20             # 定位相关文件
轮次 3:  cat src/core/parser.py                     # 阅读源码
轮次 4:  cat tests/test_parser.py                   # 阅读测试用例
轮次 5:  python -m pytest tests/test_parser.py      # 运行测试，确认 bug
轮次 6:  str_replace_editor src/core/parser.py ...  # 第一次尝试修复
轮次 7:  python -m pytest tests/test_parser.py      # 验证——仍然失败
轮次 8:  cat src/core/parser.py                     # 重新审视代码
轮次 9:  str_replace_editor src/core/parser.py ...  # 第二次修复
轮次 10: python -m pytest tests/test_parser.py      # 验证——通过！
轮次 11: submit                                     # 提交补丁
```

这种"试错 → 反思 → 再试"的循环正是 RL 训练想要教会模型的能力：**从环境反馈中学习，而不是一次做对。**

---

## 6. 快速上手

### 6.1 生成训练数据

```bash
cd /path/to/agentic-rl/verl

# 简单合成任务（流水线验证）
python recipe/swe_agent/prepare/prepare_data.py \
    --mode simple \
    --train_size 8 \
    --test_size 2 \
    --output_dir data/swe_agent_test

# SWE-bench 真实任务
python recipe/swe_agent/prepare/prepare_data.py \
    --mode swebench \
    --swebench_train /path/to/swebench_train.jsonl \
    --swebench_test /path/to/swebench_test.jsonl \
    --output_dir data/swe_agent_swebench
```

### 6.2 启动训练

```bash
cd /path/to/agentic-rl/verl

# 简单任务快速验证
bash recipe/swe_agent/example/run_qwen3_4b_instruct.sh

# SWE-bench Verified 正式训练
bash recipe/swe_agent/example/run_swebench_verified.sh
```

### 6.3 监控训练

```bash
# Ray Dashboard
# http://<HEAD_IP>:8265

# 查看训练日志
tail -f /path/to/workspace/logs/*_metrics.jsonl

# 查看 Agent 轨迹
ls /path/to/workspace/trajectories/*/rollout/
```

---

## 7. 目录结构

```
recipe/swe_agent/
├── swe_agent_loop.py              # 核心 Agent 循环（注册名 "swe_agent"）
├── config/
│   ├── swe_agent_config.yaml      # Agent 配置（模板、工具、沙箱设置）
│   ├── runtime_config.py          # 运行时配置 dataclass + 数据覆盖 + YAML 构建器
│   └── __init__.py                # 配置导出
├── runtime/
│   ├── __init__.py                # 导出 ModelProxy、execute_swe_agent 等
│   ├── model_proxy.py             # HTTP 代理：SWE-Agent ↔ vLLM
│   ├── subprocess_runner.py       # 运行 `sweagent run` 子进程 + patch 提取
│   └── container_cleanup.py       # 按 instance label 清理 Docker 容器
├── reward/
│   ├── __init__.py                # 导出 compute_score 等奖励函数
│   └── reward.py                  # 基于 patch 对比的奖励函数
├── prepare/
│   └── prepare_data.py            # 数据集生成器（简单任务 / SWE-bench）
├── utils/
│   ├── __init__.py                # 导出 PatchExtractor、消息/仓库工具
│   ├── message_utils.py           # OpenAI 消息格式归一化
│   ├── repo_manager.py            # 临时 git 仓库创建/清理
│   └── patch_extractor.py         # 从 .patch 文件或 git diff 提取补丁
├── docker/
│   └── Dockerfile.preinstalled    # 预安装 tree-sitter 依赖，避免 rollout 超时
└── example/
    ├── run_qwen3_4b_instruct.sh   # 单节点 8 GPU 快速测试脚本
    └── run_swebench_verified.sh   # SWE-bench Verified 训练脚本
```

---

## 8. 总结

| 维度 | 说明 |
|------|------|
| **解决的问题** | 让开源模型学会在真实代码环境中交互式地修复 bug |
| **核心创新** | ModelProxy 反向调用机制，实现 SWE-Agent 无侵入集成 + on-policy RL 训练 |
| **训练方式** | VERL GRPO/PPO + vLLM 推理 + Docker 沙箱，全链路分布式并行 |
| **奖励信号** | 基于 patch 对比的阶梯式奖励（0.0 ~ 1.0），支持渐进式学习 |
| **适用模型** | 已验证 Qwen3-4B-Instruct，理论上适用于任何 HuggingFace 兼容模型 |
| **任务范围** | 从简单的单文件 bug 修复到 SWE-bench 级别的真实 GitHub issue |
