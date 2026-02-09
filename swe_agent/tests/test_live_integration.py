"""
Live integration test for SWE Agent refactored components.

This test verifies that all refactored components can be instantiated
and work together in a real scenario.
"""
import pytest
import tempfile
import os
from pathlib import Path


def test_config_builder_generates_valid_file():
    """Test that ConfigBuilder can generate a valid config file."""
    from recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder

    builder = SWEAgentConfigBuilder(
        instance_id="integration-test",
        repo_path="/workspace/test",
        output_dir="/tmp/output",
        model_proxy_port=8080,
        max_steps=50,
        execution_timeout=300,
    )

    # Generate YAML
    yaml_content = builder.to_yaml()

    # Verify content
    assert "integration-test" in yaml_content
    assert "/workspace/test" in yaml_content
    assert "http://127.0.0.1:8080/v1" in yaml_content
    assert "per_instance_call_limit: 50" in yaml_content

    # Write to temporary file and verify it's valid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        import yaml
        with open(temp_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config['output_dir'] == '/tmp/output'
        assert config['problem_statement']['id'] == 'integration-test'
    finally:
        os.unlink(temp_path)


def test_config_validator_integration():
    """Test that ConfigValidator properly validates configs."""
    from recipe.swe_agent.config.config_validator import (
        AgentConfigValidator,
        ConfigValidationError
    )

    # Valid config
    valid_config = {
        "proxy_config": {"port": 8080},
        "sandbox_config": {
            "swe_agent_timeout": 1800,
            "max_steps": 100,
        },
    }

    validator = AgentConfigValidator(valid_config)
    validator.validate()  # Should not raise

    # Invalid config
    invalid_config = {
        "proxy_config": {"port": 99999},  # Invalid port
        "sandbox_config": {},
    }

    validator = AgentConfigValidator(invalid_config)
    with pytest.raises(ConfigValidationError):
        validator.validate()


@pytest.mark.asyncio
async def test_patch_extractor_with_real_git():
    """Test PatchExtractor with a real git repository."""
    from recipe.swe_agent.utils.patch_extractor import PatchExtractor

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        os.system(f"cd {repo_path} && git init && git config user.email 'test@test.com' && git config user.name 'Test' 2>/dev/null")

        # Create and commit a file
        test_file = repo_path / "hello.py"
        test_file.write_text("print('hello world')")
        os.system(f"cd {repo_path} && git add . && git commit -m 'init' 2>/dev/null")

        # Modify the file
        test_file.write_text("print('hello refactored world!')")

        # Extract patch
        extractor = PatchExtractor(
            output_dir=str(tmpdir),
            instance_id="test",
            repo_path=str(repo_path),
        )

        patch = await extractor.extract()

        # Verify patch
        assert patch is not None
        assert "hello.py" in patch
        assert "refactored" in patch or "-hello world" in patch


def test_swe_agent_loop_can_import():
    """Test that SWEAgentLoop can be imported with all refactored dependencies."""
    try:
        from recipe.swe_agent.swe_agent_loop import SWEAgentLoop

        # Verify it's a class
        assert isinstance(SWEAgentLoop, type)

        # Verify it has the expected methods
        assert hasattr(SWEAgentLoop, '_validate_config')
        assert hasattr(SWEAgentLoop, '_generate_swe_agent_config')
        assert hasattr(SWEAgentLoop, '_extract_patch')

        print("✅ SWEAgentLoop successfully imported with all refactored components")

    except Exception as e:
        pytest.fail(f"Failed to import SWEAgentLoop: {e}")


def test_docker_sandbox_can_import():
    """Test that DockerSandbox can be imported with all refactored dependencies."""
    try:
        from recipe.swe_agent.sandbox.docker_sandbox import DockerSandbox

        # Verify it's a class
        assert isinstance(DockerSandbox, type)

        # Verify it has the expected methods
        assert hasattr(DockerSandbox, '_generate_swe_agent_config')
        assert hasattr(DockerSandbox, 'get_patch')
        assert hasattr(DockerSandbox, 'create')

        print("✅ DockerSandbox successfully imported with all refactored components")

    except Exception as e:
        pytest.fail(f"Failed to import DockerSandbox: {e}")


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
