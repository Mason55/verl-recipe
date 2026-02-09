"""
End-to-end smoke test for refactored SWE Agent components.

This test verifies that all refactored components work together
without requiring GPU or full training infrastructure.
"""
import tempfile
import os
from pathlib import Path


def test_config_builder_produces_valid_swe_agent_config():
    """Verify ConfigBuilder produces a valid SWE-Agent compatible config."""
    from recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder

    builder = SWEAgentConfigBuilder(
        instance_id="smoke-test-001",
        repo_path="/workspace/test-repo",
        output_dir="/tmp/swe-agent-output",
        model_proxy_port=8080,
        max_steps=50,
        execution_timeout=300,
    )

    # Generate config
    config_dict = builder.build()

    # Verify critical fields exist
    assert config_dict["output_dir"] == "/tmp/swe-agent-output"
    assert config_dict["problem_statement"]["id"] == "smoke-test-001"
    assert config_dict["env"]["repo"]["path"] == "/workspace/test-repo"
    assert config_dict["agent"]["model"]["api_base"] == "http://127.0.0.1:8080/v1"
    assert config_dict["agent"]["model"]["per_instance_call_limit"] == 50
    assert config_dict["agent"]["tools"]["execution_timeout"] == 300

    # Verify templates are included
    assert "system_template" in config_dict["agent"]["templates"]
    assert "instance_template" in config_dict["agent"]["templates"]

    # Verify tool bundles are configured
    assert len(config_dict["agent"]["tools"]["bundles"]) > 0

    print("✅ ConfigBuilder produces valid SWE-Agent config")


def test_config_validator_catches_errors():
    """Verify ConfigValidator properly validates configurations."""
    from recipe.swe_agent.config.config_validator import (
        AgentConfigValidator,
        ConfigValidationError
    )
    import pytest

    # Test valid config passes
    valid_config = {
        "proxy_config": {"port": 8080},
        "sandbox_config": {
            "swe_agent_timeout": 1800,
            "max_steps": 100,
        },
    }

    validator = AgentConfigValidator(valid_config)
    validator.validate()  # Should not raise

    # Test invalid port is caught
    invalid_port_config = {
        "proxy_config": {"port": 999999},  # Invalid
        "sandbox_config": {},
    }

    validator = AgentConfigValidator(invalid_port_config)
    try:
        validator.validate()
        assert False, "Should have raised ConfigValidationError"
    except ConfigValidationError:
        pass  # Expected

    print("✅ ConfigValidator properly validates configs")


def test_patch_extractor_strategies():
    """Verify PatchExtractor tries multiple strategies correctly."""
    import asyncio
    from recipe.swe_agent.utils.patch_extractor import PatchExtractor

    async def run_test():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test 1: Extract from .patch file
            patch_dir = Path(tmpdir) / "test-001"
            patch_dir.mkdir()
            patch_file = patch_dir / "test-001.patch"
            expected_content = "diff --git a/file.py b/file.py\n+new line"
            patch_file.write_text(expected_content)

            extractor = PatchExtractor(
                output_dir=tmpdir,
                instance_id="test-001",
                repo_path=None,
            )

            patch = await extractor.extract()
            assert patch == expected_content

            # Test 2: No patch file, no git repo -> returns None
            extractor2 = PatchExtractor(
                output_dir="/tmp/nonexistent",
                instance_id="test-002",
                repo_path=None,
            )

            patch2 = await extractor2.extract()
            assert patch2 is None

    asyncio.run(run_test())
    print("✅ PatchExtractor strategies work correctly")


def test_all_components_integrate():
    """
    Integration test: Verify all refactored components can be used together
    in a workflow similar to actual SWEAgentLoop.
    """
    from recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder
    from recipe.swe_agent.config.config_validator import AgentConfigValidator
    from recipe.swe_agent.utils.patch_extractor import PatchExtractor
    import asyncio

    async def run_integration():
        # Step 1: Validate config (like SWEAgentLoop.__init__)
        agent_config = {
            "proxy_config": {"port": 8080},
            "sandbox_config": {
                "swe_agent_timeout": 1800,
                "max_steps": 100,
            },
        }

        validator = AgentConfigValidator(agent_config)
        validator.validate()

        # Step 2: Generate SWE-Agent config (like _generate_swe_agent_config)
        builder = SWEAgentConfigBuilder(
            instance_id="integration-test",
            repo_path="/workspace/repo",
            output_dir="/tmp/output",
            model_proxy_port=agent_config["proxy_config"]["port"],
            max_steps=agent_config["sandbox_config"]["max_steps"],
            execution_timeout=300,
        )

        config_yaml = builder.to_yaml()
        assert len(config_yaml) > 0
        assert "integration-test" in config_yaml

        # Step 3: Extract patch (like _extract_patch)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate SWE-Agent creating a patch file
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            patch_file = output_dir / "result.patch"
            test_patch = "diff --git a/main.py b/main.py\n+print('refactored!')"
            patch_file.write_text(test_patch)

            extractor = PatchExtractor(
                output_dir=str(output_dir),
                instance_id="integration-test",
                repo_path=None,
            )

            patch = await extractor.extract()
            assert patch == test_patch

    asyncio.run(run_integration())
    print("✅ All components integrate successfully")


def main():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("SWE Agent Refactoring - End-to-End Smoke Tests")
    print("="*60 + "\n")

    test_config_builder_produces_valid_swe_agent_config()
    test_config_validator_catches_errors()
    test_patch_extractor_strategies()
    test_all_components_integrate()

    print("\n" + "="*60)
    print("✅✅✅ ALL SMOKE TESTS PASSED ✅✅✅")
    print("="*60)
    print("\nRefactored components are ready for production use:")
    print("  ✅ SWEAgentConfigBuilder - Unified config generation")
    print("  ✅ AgentConfigValidator - Config validation")
    print("  ✅ PatchExtractor - Multi-strategy patch extraction")
    print("  ✅ Integration - All components work together")
    print()


if __name__ == "__main__":
    main()
