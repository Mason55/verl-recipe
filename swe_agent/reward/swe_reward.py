# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SWE Agent Reward Function

基于 patch 的正确性计算奖励：
1. 简单模式：比较生成的 patch 与 expected_patch
2. SWE-bench 模式：运行测试用例验证 patch

奖励计算：
- 完全正确：1.0
- 部分正确：0.5
- 生成了有效 patch 但不正确：0.1
- 无法生成 patch：0.0
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize_patch(patch: str) -> str:
    """规范化 patch 字符串，便于比较。
    
    Args:
        patch: 原始 patch 字符串
        
    Returns:
        规范化后的 patch
    """
    if not patch:
        return ""
    
    # 移除空白行
    lines = [line.rstrip() for line in patch.strip().split("\n")]
    lines = [line for line in lines if line]
    
    # 移除 git diff 头部中的 commit hash
    normalized_lines = []
    for line in lines:
        # 跳过 index 行
        if line.startswith("index "):
            continue
        # 跳过空行
        if not line.strip():
            continue
        normalized_lines.append(line)
    
    return "\n".join(normalized_lines)


def extract_changed_files(patch: str) -> List[str]:
    """从 patch 中提取修改的文件列表。
    
    Args:
        patch: patch 字符串
        
    Returns:
        文件路径列表
    """
    if not patch:
        return []
    
    files = []
    # 匹配 diff --git a/path b/path
    pattern = r"diff --git a/(.+?) b/(.+)"
    matches = re.findall(pattern, patch)
    for match in matches:
        files.append(match[1])  # 使用 b/ 路径
    
    return files


def compare_patches_simple(generated: str, expected: str) -> float:
    """简单比较两个 patch。
    
    Args:
        generated: 生成的 patch
        expected: 期望的 patch
        
    Returns:
        相似度分数 (0.0 - 1.0)
    """
    if not generated:
        return 0.0
    
    gen_normalized = normalize_patch(generated)
    exp_normalized = normalize_patch(expected)
    
    # 完全匹配
    if gen_normalized == exp_normalized:
        return 1.0
    
    # 比较修改的文件
    gen_files = set(extract_changed_files(generated))
    exp_files = set(extract_changed_files(expected))
    
    if not exp_files:
        return 0.1 if gen_files else 0.0
    
    # 文件匹配度
    file_overlap = len(gen_files & exp_files) / len(exp_files)
    
    if file_overlap == 1.0:
        # 文件完全匹配，给予部分分数
        return 0.5
    elif file_overlap > 0:
        # 部分文件匹配
        return 0.2 + 0.3 * file_overlap
    else:
        # 没有文件匹配，但生成了 patch
        return 0.1


def compute_swe_agent_reward(
    generated_patch: Optional[str],
    reward_model_config: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """计算 SWE Agent 的奖励。
    
    Args:
        generated_patch: 生成的 patch
        reward_model_config: 奖励模型配置
        extra_info: 额外信息
        
    Returns:
        奖励分数 (0.0 - 1.0)
    """
    style = reward_model_config.get("style", "swe_agent")
    
    if style == "swe_agent":
        # 简单模式：直接比较 patch
        expected_patch = reward_model_config.get("ground_truth", "")
        return compare_patches_simple(generated_patch, expected_patch)
    
    elif style == "swe_bench":
        # SWE-bench 模式：需要运行测试
        # 这里先使用简单比较，后续可以集成实际的测试运行
        gold_patch = reward_model_config.get("gold_patch", "")
        
        if not generated_patch:
            return 0.0
        
        # 简单比较
        score = compare_patches_simple(generated_patch, gold_patch)
        
        # TODO: 集成 SWE-bench 的测试运行
        # 这需要：
        # 1. 克隆仓库到指定 commit
        # 2. 应用生成的 patch
        # 3. 运行测试用例
        # 4. 根据测试结果计算奖励
        
        return score
    
    else:
        logger.warning(f"Unknown reward style: {style}, using default comparison")
        ground_truth = reward_model_config.get("ground_truth", "")
        return compare_patches_simple(generated_patch, ground_truth)


def compute_batch_rewards(
    generated_patches: List[Optional[str]],
    reward_model_configs: List[Dict[str, Any]],
    extra_infos: Optional[List[Dict[str, Any]]] = None,
) -> List[float]:
    """批量计算奖励。
    
    Args:
        generated_patches: 生成的 patch 列表
        reward_model_configs: 奖励模型配置列表
        extra_infos: 额外信息列表
        
    Returns:
        奖励分数列表
    """
    if extra_infos is None:
        extra_infos = [None] * len(generated_patches)
    
    rewards = []
    for patch, config, info in zip(generated_patches, reward_model_configs, extra_infos):
        reward = compute_swe_agent_reward(patch, config, info)
        rewards.append(reward)
    
    return rewards


# 测试代码
if __name__ == "__main__":
    # 测试简单比较
    expected = """diff --git a/1.txt b/2.txt
similarity index 100%
rename from 1.txt
rename to 2.txt"""
    
    generated_correct = expected
    generated_partial = """diff --git a/1.txt b/2.txt
--- a/1.txt
+++ b/2.txt"""
    generated_wrong = """diff --git a/other.txt b/other.txt
--- a/other.txt
+++ b/other.txt"""
    
    config = {"style": "swe_agent", "ground_truth": expected}
    
    print("Testing SWE Agent Reward Function:")
    print(f"Correct patch: {compute_swe_agent_reward(generated_correct, config)}")  # Should be 1.0
    print(f"Partial patch: {compute_swe_agent_reward(generated_partial, config)}")  # Should be ~0.5
    print(f"Wrong patch: {compute_swe_agent_reward(generated_wrong, config)}")  # Should be ~0.1
    print(f"Empty patch: {compute_swe_agent_reward(None, config)}")  # Should be 0.0
