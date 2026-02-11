# SWE-Agent VERL 训练指标说明文档

本文档详细解释 `plot_training_curves.py` 中绘制的 9 张训练曲线图涉及的所有指标含义，帮助理解 GRPO (Group Relative Policy Optimization) 强化学习训练过程中的关键信号。

---

## 概览：9 张子图一览

| 位置 | 图表名称 | 核心问题 | 健康信号 |
|------|----------|----------|----------|
| (0,0) | Training Reward (Score) | 模型在训练集上做对了多少？ | 随训练稳步上升 |
| (0,1) | Validation Reward & Accuracy | 泛化到未见任务的效果如何？ | 随训练上升，不过拟合 |
| (0,2) | Policy Gradient Loss | 策略梯度方向是否有效？ | 负值且趋于稳定 |
| (1,0) | Policy Entropy | 模型是否还在探索？ | 缓慢下降，不能骤降到 0 |
| (1,1) | KL Divergence | 新策略偏离参考策略多远？ | 适度增长，不能爆炸 |
| (1,2) | Gradient Norm | 梯度更新幅度是否稳定？ | 保持在合理范围，无尖刺 |
| (2,0) | Response Length | 模型输出越来越长还是越来越精炼？ | 趋于稳定或适度缩短 |
| (2,1) | Number of Turns | Agent 交互轮数是否合理？ | 随训练逐渐减少 (更高效) |
| (2,2) | PPO Clip Fraction | 策略更新被裁剪的比例？ | 低但非零 (5-15% 健康) |

---

## 第一行：奖励与损失

### 1. Training Reward (Score) — 训练奖励分数

```
指标键: critic/score/mean, critic/score/max, critic/score/min
```

**含义**：每个训练 step 中，一个 batch 内所有样本的**奖励分数统计**。

- **Mean (均值)**：batch 中所有样本奖励的平均值，是**最核心的训练信号**
- **Max (最大值)**：batch 中得分最高的样本
- **Min (最小值)**：batch 中得分最低的样本
- **阴影区域**：Min 到 Max 的范围，反映 batch 内分数的离散程度

**分数来源**：由 `reward/reward.py` 中的 `compare_patches_simple()` 计算：

| 分数 | 含义 |
|------|------|
| **1.0** | Patch 完全匹配 (标准化后) |
| **0.5** | 修改的文件全部命中，但内容不完全一致 |
| **0.2 ~ 0.5** | 部分文件命中 |
| **0.1** | 生成了 patch 但文件完全不对 |
| **0.0** | 没有生成 patch |

**怎么看**：
- 理想情况：从 0.3-0.5 逐步上升到 0.7+
- 如果长期停在 0.1 附近 → 模型学会了生成 patch 的格式但内容不对
- 如果 Max 一直是 1.0 但 Mean 不涨 → 只有简单任务做对了

---

### 2. Validation Reward & Accuracy — 验证集奖励与准确率

```
指标键: val-aux/swe_agent_simple/reward/mean@1
       val-core/swe_agent_simple/acc/mean@1
```

**含义**：在**未参与训练的验证集**上评估模型性能。

- **Val Reward (验证奖励)**：验证集所有样本的平均奖励分数
- **Val Accuracy (验证准确率)**：验证集准确率 (同 reward，因为此处 acc = reward)
- `@1` 表示每个 prompt 只采样 1 次 (`n_resp_per_prompt_val=1`)

**怎么看**：
- 训练集分数上升 + 验证集也上升 → **正常收敛**
- 训练集上升 + 验证集停滞/下降 → **过拟合**
- 两者都不动 → 学习率太小或数据太难

**与训练分数的区别**：验证集使用 `do_sample=False` (greedy decoding)，所以结果更稳定、可复现。

---

### 3. Policy Gradient Loss — 策略梯度损失

```
指标键: actor/pg_loss
```

**含义**：GRPO 的策略梯度损失函数值，衡量当前策略更新的方向和幅度。

**公式本质**：
```
PG Loss = -E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ]
```
其中：
- `r(θ)` = 新策略概率 / 旧策略概率 (importance sampling ratio)
- `A` = 优势值 (advantage，由 GRPO 中的 group-relative baseline 计算)
- `ε` = clip 范围 (本项目中 `clip_ratio_low=0.2, clip_ratio_high=0.28`)

**怎么看**：
- **负值是正常的**：因为公式取负号，负值越大表示策略在往高奖励方向更新
- 稳步下降 (更负) → 策略持续改进
- 剧烈波动 → 学习不稳定，考虑降低学习率
- 接近 0 → 策略不再变化 (可能收敛也可能陷入局部最优)

---

## 第二行：策略行为分析

### 4. Policy Entropy — 策略熵

```
指标键: actor/entropy
```

**含义**：策略输出的 token 概率分布的**信息熵**，衡量模型输出的**随机性/多样性**。

- **高熵** → 模型在各 token 上的概率比较均匀，输出多样 (探索多)
- **低熵** → 模型非常确定地选择少数 token，输出集中 (利用多)

**怎么看**：
- 缓慢下降 → **正常**，模型学到了更确定的策略
- 骤降至接近 0 → **模式坍塌 (mode collapse)**，模型只会生成固定模板，很危险
- 完全不变 → 模型没有学到新信息
- 上升 → 模型变得更不确定 (可能奖励信号矛盾)

**经验值**：对于 Qwen3-4B 这类模型，合理范围约 0.1 ~ 0.5。

---

### 5. KL Divergence — KL 散度

```
指标键: rollout_corr/kl    (Rollout KL)
       actor/ppo_kl        (PPO KL)
```

**含义**：衡量**当前策略与参考策略 (初始模型)** 之间的偏离程度。

- **Rollout KL**：基于 rollout (采样) 时的概率比计算的 KL 散度
  - 公式：`KL = E[log(π_new / π_ref)]`
  - 反映的是序列级别的策略偏移

- **PPO KL**：PPO 更新步骤中计算的 KL 散度
  - 基于 actor 更新前后概率比的变化
  - 更侧重于单步更新的变化量

**怎么看**：
- 缓慢上升 → **正常**，策略在逐渐远离初始模型
- 快速爆炸 → **危险**，策略跑偏了，需要增大 KL 惩罚 (`kl_coef`)
- 接近 0 → 策略几乎没变化

> 注意：本项目中 `use_kl_in_reward=false, kl_coef=0.0`，即**不使用 KL 惩罚**。GRPO 通过 clip 机制隐式约束策略偏移，所以 KL 可以只作为监控指标。

---

### 6. Gradient Norm — 梯度范数

```
指标键: actor/grad_norm
```

**含义**：每次参数更新时，所有梯度的 L2 范数，衡量**参数更新的幅度**。

**怎么看**：
- 稳定在某个范围内 → **正常**
- 持续上升 → 训练可能不稳定，考虑：
  - 降低学习率 (`actor_lr`)
  - 启用梯度裁剪
- 出现尖刺 → 某个 batch 的梯度异常大 (可能遇到了极端样本)
- 接近 0 → 梯度消失，模型不再学习

**经验值**：通常在 1.0 ~ 10.0 范围内比较健康。

---

## 第三行：Agent 行为分析

### 7. Response Length — 响应长度 (tokens)

```
指标键: response_length/mean, response_length/max, response_length/min
```

**含义**：模型每次生成的**响应 token 数**统计。对于 SWE-Agent 场景，这是整个多轮交互中所有模型回复的**总 token 数**。

- **Mean**：batch 内平均响应长度
- **Max / Min**：最长/最短的响应
- **阴影区域**：Min 到 Max 的范围

**怎么看**：
- 逐渐缩短 → 模型学会了更高效地解决问题 (更少废话)，**好信号**
- 逐渐变长 → 模型可能在"注水"，用长输出换取概率奖励，**需警惕**
- 突然骤降 → 可能模式坍塌，模型只输出固定短句
- 始终打满 `max_response_length` → 任务太难，或 max_response_length 设置太小

---

### 8. Number of Turns — 交互轮数

```
指标键: num_turns/mean, num_turns/max, num_turns/min   (训练)
       val-aux/num_turns/mean                            (验证)
```

**含义**：SWE-Agent 完成一个任务所用的**对话轮数** (每轮 = 一次 LLM 调用 + 一次工具执行)。

- **Train Mean (训练均值)**：训练 batch 中的平均轮数
- **Val Mean (验证均值)**：验证集上的平均轮数
- **阴影区域**：Min 到 Max 的范围

**怎么看**：
- 逐渐减少 → 模型学会了更高效地完成任务，**非常好的信号**
- 始终打满 max_turns → 任务太难，或模型还没学会正确的工具使用模式
- 训练轮数减少但验证轮数不变 → 在训练集上过拟合了特定任务模式

> 对于简单的文件编辑任务，理想轮数是 2-3 轮 (探索 → 修改 → 提交)。

---

### 9. PPO Clip Fraction — PPO 裁剪比例

```
指标键: actor/pg_clipfrac
```

**含义**：在 PPO/GRPO 更新中，importance sampling ratio `r(θ)` 被裁剪的**比例**。

**机制解释**：
```
clip(r(θ), 1 - ε_low, 1 + ε_high)
```
当 `r(θ)` 超出 `[1-0.2, 1+0.28]` = `[0.8, 1.28]` 范围时，梯度会被裁剪。clip fraction 就是被裁剪的 token 占总 token 的比例。

**怎么看**：
- **0 ~ 5%** → 策略更新很保守，几乎没有被裁剪
- **5 ~ 15%** → **健康范围**，说明策略在积极更新但不过度
- **> 30%** → 策略更新过于激进，新旧策略差异太大
  - 可能原因：学习率太高、rollout 与训练之间延迟太长
- **持续 0%** → 策略没有实质性更新

---

## 补充指标 (未绘图但日志中可见)

以下指标出现在 metrics JSONL 中但未单独绘图，可用于深入分析：

| 指标 | 含义 |
|------|------|
| `actor/lr` | 当前学习率 |
| `actor/kl_loss` | KL 损失 (本项目为 0，因为 `use_kl_loss=false`) |
| `actor/pg_clipfrac_lower` | 下界裁剪比例 (ratio < 1-ε) |
| `critic/rewards/mean` | 同 score/mean (GRPO 中 reward = score) |
| `critic/advantages/mean` | GRPO 优势值均值 (相对组内 baseline) |
| `critic/returns/mean` | 同 advantages (GRPO 无 value network) |
| `prompt_length/mean` | 输入 prompt 长度 |
| `response/aborted_ratio` | 响应被中止的比例 (超长截断) |
| `rollout_corr/training_ppl` | 训练时困惑度 |
| `rollout_corr/ppl_ratio` | 困惑度比率 (新旧策略) |
| `perf/throughput` | 训练吞吐量 (tokens/s) |
| `timing_s/gen` | 采样/生成耗时 (秒) |
| `timing_s/update_actor` | Actor 更新耗时 (秒) |

---

## 轨迹文件说明

### Rollout 轨迹 (`rollout_data_dir`)

每个 step 输出一个 `{step}.jsonl`，每行一个 JSON 对象：

```json
{
    "input": "system\nYou are a helpful assistant...",   // 完整 prompt
    "output": "DISCUSSION\nThe task is to...",           // 模型完整输出
    "gts": "diff --git a/1.txt b/2.txt...",             // 期望 patch
    "score": 1.0,                                        // 奖励分数
    "step": 1,                                           // 训练 step
    "acc": 1.0                                           // 准确率
}
```

### Validation 轨迹 (`validation_data_dir`)

格式与 Rollout 相同，但使用 greedy decoding，每 `test_freq` 步输出一次。

---

## 如何判断训练是否正常收敛？

### 正常收敛的典型模式：

1. **Training Score** 从 ~0.3 逐步上升至 ~0.7+
2. **Validation Score** 跟随上升，差距不超过 0.1-0.2
3. **Entropy** 从 ~0.3 缓慢下降至 ~0.15
4. **PG Loss** 负值增大后趋于稳定
5. **Grad Norm** 在 2-8 范围内波动
6. **Turns** 逐渐减少 (从 ~5 降至 ~3)
7. **Clip Fraction** 维持在 5-15%

### 常见异常及应对：

| 异常现象 | 可能原因 | 应对措施 |
|----------|----------|----------|
| Score 停在 0.0-0.1 | 模型没学会生成 patch | 检查 agent loop 配置，增加 max_turns |
| Entropy 骤降到 0 | 模式坍塌 | 降低学习率，增加 KL 惩罚 |
| Grad Norm 爆炸 (>50) | 学习不稳定 | 降低 `actor_lr`，加梯度裁剪 |
| Clip Fraction > 30% | 策略更新过于激进 | 降低学习率或缩小 clip 范围 |
| Val Score 下降 | 过拟合 | 增加数据多样性，减少 epochs |
| KL 快速增长 | 策略跑偏 | 开启 `use_kl_in_reward=true` |
