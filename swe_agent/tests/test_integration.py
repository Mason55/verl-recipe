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
Integration tests for SWE Agent Loop.

These tests verify the full flow with mocked components.
"""
import pytest


@pytest.mark.asyncio
async def test_subprocess_mode_end_to_end():
    """Test subprocess execution mode end-to-end."""
    # This is a placeholder for integration testing
    # Real implementation would mock server_manager and validate flow
    pass


@pytest.mark.asyncio
async def test_docker_mode_end_to_end():
    """Test Docker execution mode end-to-end."""
    # This is a placeholder for integration testing
    # Real implementation would mock Docker and validate flow
    pass


def test_config_builder_integration():
    """Test that ConfigBuilder integrates correctly."""
    from recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder

    builder = SWEAgentConfigBuilder(
        instance_id="test",
        repo_path="/workspace",
        output_dir="/tmp",
        model_proxy_port=8080,
    )

    yaml_str = builder.to_yaml()
    assert "test" in yaml_str
    assert "/workspace" in yaml_str


@pytest.mark.asyncio
async def test_patch_extractor_integration():
    """Test that PatchExtractor integrates correctly."""
    from recipe.swe_agent.utils.patch_extractor import PatchExtractor

    extractor = PatchExtractor(
        output_dir="/tmp",
        instance_id="test",
        repo_path=None,
    )

    # Should return None gracefully when no patch exists
    patch = await extractor.extract()
    assert patch is None
