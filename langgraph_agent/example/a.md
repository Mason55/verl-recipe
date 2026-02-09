# SWE-bench Verified 数据集字段全解析（SWE-bench/SWE-bench_Verified）

以下是SWE-bench Verified数据集中**所有字段的详细含义**，基于官方Hugging Face仓库文档和样本实例整理，按逻辑分组呈现，帮助你全面理解数据集结构。

---

## 一、核心标识字段（定位与版本控制）

| 字段名 | 数据类型 | 核心含义 | 示例 |
|--------|----------|----------|------|
| **instance_id** | str | 唯一实例标识符，格式为 `repo_owner__repo_name-PR-number`，用于区分不同任务 | `astropy__astropy-13319` |
| **repo** | str | GitHub仓库完整路径（所有者/仓库名），表示问题来源项目 | `astropy/astropy` |
| **base_commit** | str | 修复前代码库的基准提交哈希，代表问题存在时的代码状态 | `1a2b3c4d5e6f7g8h9i0j` |
| **environment_setup_commit** | str | 用于环境配置和安装的提交哈希，确保评估环境一致性 | `9z8y7x6w5v4u3t2s1r0q` |
| **version** | str | 评估时使用的安装版本号，指定依赖环境版本 | `5.1.dev` |
| **created_at** | str | PR创建时间，用于时间戳记录和数据分析 | `2023-05-15T14:30:00Z` |

---

## 二、问题与解决方案核心字段（任务内容）

| 字段名 | 数据类型 | 核心含义 | 关键作用 |
|--------|----------|----------|----------|
| **problem_statement** | str | 完整的问题描述，包含issue标题和正文，是模型需要解决的任务 | 作为模型输入，描述需要修复的bug或实现的功能 |
| **patch** | str | 黄金修复补丁（diff格式），不含测试相关代码，是PR解决问题的核心代码变更 | 1. 用于对比模型生成的补丁<br>2. 评估时作为参考解决方案 |
| **test_patch** | str | PR中贡献的测试文件补丁（diff格式），包含新增/修改的测试用例 | 1. 标记哪些测试与问题修复相关<br>2. 评估时用于验证模型修复的有效性 |
| **hints_text** | str | 问题创建到PR首次提交之间的所有issue评论，可能包含额外上下文或线索 | 提供问题解决的额外信息，可用于检索增强或提示工程 |

---

## 三、评估关键字段（测试验证标准）

这是SWE-bench评估机制的核心，决定了模型修复是否被判定为成功。

| 字段名 | 数据类型 | 核心含义 | 评估逻辑 |
|--------|----------|----------|----------|
| **FAIL_TO_PASS** | JSON字符串（列表） | PR应用前后从**失败→通过**的测试用例列表，是修复有效性的核心验证指标 | 模型修复必须让这些测试全部通过，否则视为修复失败 |
| **PASS_TO_PASS** | JSON字符串（列表） | PR应用前后**始终通过**的测试用例列表，用于验证修复没有引入新bug | 模型修复不能导致这些测试失败，确保修复的安全性和兼容性 |

### 评估原理说明
SWE-bench采用**单元测试验证**机制：
1. 在`base_commit`状态下运行测试，确认`FAIL_TO_PASS`测试失败，`PASS_TO_PASS`测试通过
2. 应用模型生成的补丁
3. 重新运行测试，只有当**所有`FAIL_TO_PASS`测试通过**且**所有`PASS_TO_PASS`测试保持通过**时，修复才被判定为成功

---

## 四、字段分组与使用场景指南

### 1. 模型输入字段（用于问题理解）
- **problem_statement**：核心输入，包含完整的bug报告或功能请求
- **hints_text**：可选补充输入，提供问题讨论中的额外线索
- **repo + base_commit**：用于定位代码库状态，获取完整上下文

### 2. 评估参考字段（用于结果验证）
- **patch**：黄金标准，用于对比模型生成的补丁（可选，非评估必需）
- **test_patch**：用于识别与问题相关的测试文件
- **FAIL_TO_PASS + PASS_TO_PASS**：评估核心标准，决定模型修复是否有效
- **environment_setup_commit + version**：确保评估环境与原始修复一致

### 3. 管理与分析字段
- **instance_id**：唯一标识，用于结果跟踪和排行榜提交
- **created_at**：用于分析问题解决的时间分布和趋势
- **repo**：用于按项目分组评估，分析模型在不同类型项目上的表现差异

---

## 五、字段使用注意事项

1. **数据类型转换**：
   - `FAIL_TO_PASS`和`PASS_TO_PASS`存储为JSON字符串，使用前需用`json.loads()`转换为Python列表
   ```python
   import json
   fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
   pass_to_pass = json.loads(instance["PASS_TO_PASS"])
   ```

2. **评估独立性**：
   - 评估不依赖`patch`字段，仅通过测试结果判定，避免模型直接复制黄金补丁的作弊行为
   - `patch`主要用于研究分析和模型训练参考

3. **环境一致性**：
   - `environment_setup_commit`和`version`是确保评估可复现的关键，必须严格遵循，否则可能导致测试结果不稳定

---

## 六、完整字段速查表

| 字段名 | 类型 | 用途分类 | 核心价值 |
|--------|------|----------|----------|
| instance_id | str | 管理 | 唯一任务标识 |
| repo | str | 输入+评估 | 代码库定位 |
| base_commit | str | 输入+评估 | 代码库状态 |
| problem_statement | str | 输入 | 完整问题描述 |
| hints_text | str | 输入 | 额外讨论线索 |
| patch | str | 参考 | 黄金修复补丁 |
| test_patch | str | 评估 | 相关测试文件 |
| FAIL_TO_PASS | JSON str | 评估 | 核心测试标准（失败→通过） |
| PASS_TO_PASS | JSON str | 评估 | 核心测试标准（保持通过） |
| environment_setup_commit | str | 评估 | 环境一致性 |
| version | str | 评估 | 版本控制 |
| created_at | str | 分析 | 时间戳 |

---

## 总结
SWE-bench Verified的字段设计遵循**真实软件工程流程**，从问题理解（problem_statement）→ 代码修改 → 测试验证（FAIL_TO_PASS + PASS_TO_PASS），完整模拟了开发者解决GitHub问题的全过程。每个字段都有明确的用途，既支持模型输入，又确保评估的客观性和可复现性，是当前AI软件工程能力评估的黄金标准。