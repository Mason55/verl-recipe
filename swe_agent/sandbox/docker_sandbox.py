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
Docker Sandbox for SWE-Agent

提供 Docker 容器隔离环境，用于安全地运行 SWE-Agent。

主要功能：
1. 容器生命周期管理（创建、启动、停止、清理）
2. 文件系统隔离
3. 网络配置（允许 SWE-Agent 访问 ModelProxy）
4. 执行命令和获取输出
"""

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from verl.recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder
from verl.recipe.swe_agent.utils.patch_extractor import PatchExtractor

logger = logging.getLogger(__name__)


@dataclass
class DockerSandboxConfig:
    """Docker Sandbox 配置"""
    
    # 镜像配置
    image: str = "verl-swe-agent:latest"
    
    # 容器资源限制
    memory_limit: str = "8g"
    cpu_count: int = 4
    
    # 超时配置
    startup_timeout: int = 60  # 容器启动超时
    execution_timeout: int = 1800  # SWE-Agent 执行超时
    
    # 网络配置
    network_mode: str = "host"  # 使用 host 网络，便于访问 ModelProxy
    
    # 工作目录
    work_dir: str = "/workspace"
    
    # ModelProxy 配置
    model_proxy_host: str = "127.0.0.1"
    model_proxy_port: int = 8080
    
    # 额外环境变量
    env_vars: Dict[str, str] = field(default_factory=dict)


class DockerSandbox:
    """Docker 容器沙箱，用于隔离运行 SWE-Agent"""
    
    def __init__(self, config: Optional[DockerSandboxConfig] = None):
        """初始化 Docker Sandbox
        
        Args:
            config: 沙箱配置，如果为 None 则使用默认配置
        """
        self.config = config or DockerSandboxConfig()
        self.container_id: Optional[str] = None
        self.container_name: Optional[str] = None
        self._temp_dir: Optional[str] = None
        self._docker_available: Optional[bool] = None
        
    async def _check_docker_available(self) -> bool:
        """检查 Docker 是否可用"""
        if self._docker_available is not None:
            return self._docker_available
            
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            self._docker_available = process.returncode == 0
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self._docker_available = False
            
        return self._docker_available
    
    async def _check_image_exists(self) -> bool:
        """检查 Docker 镜像是否存在"""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "image", "inspect", self.config.image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=30)
            return process.returncode == 0
        except Exception:
            return False
    
    async def create(self, repo_content: Optional[Dict[str, str]] = None) -> str:
        """创建并启动容器

        Args:
            repo_content: 初始仓库内容，格式为 {filename: content}

        Returns:
            容器 ID

        Raises:
            RuntimeError: Docker 不可用、镜像不存在或容器创建失败
        """
        # Step 1: Validate environment
        await self._ensure_docker_available()

        # Step 2: Ensure image exists
        await self._ensure_image_available()

        # Step 3: Prepare workspace
        self._temp_dir = tempfile.mkdtemp(prefix="swe_agent_")
        if repo_content:
            await self._init_repo(repo_content)

        # Step 4: Create and start container
        self.container_name = f"swe-agent-{uuid.uuid4().hex[:8]}"
        self.container_id = await self._create_container()

        logger.info(f"Container ready: {self.container_id[:12]}")
        return self.container_id

    async def _ensure_docker_available(self) -> None:
        """确保 Docker 可用

        Raises:
            RuntimeError: Docker 不可用时抛出，包含详细的问题排查指南
        """
        if not await self._check_docker_available():
            raise RuntimeError(
                "Docker is not available. Please ensure:\n"
                "1. Docker is installed (run: docker --version)\n"
                "2. Docker daemon is running (run: docker ps)\n"
                "3. If running in container, mount Docker socket:\n"
                "   -v /var/run/docker.sock:/var/run/docker.sock\n"
                "4. Docker binary is accessible:\n"
                "   -v /usr/bin/docker:/usr/bin/docker"
            )

    async def _ensure_image_available(self) -> None:
        """确保 Docker 镜像可用

        Raises:
            RuntimeError: 镜像不存在且构建失败时抛出
        """
        if not await self._check_image_exists():
            logger.warning(f"Image {self.config.image} not found, attempting to build...")
            await self._build_image()

    async def _create_container(self) -> str:
        """创建并启动 Docker 容器

        Returns:
            容器 ID

        Raises:
            RuntimeError: 容器创建失败或启动超时
        """
        cmd = self._build_docker_run_cmd()

        logger.info(f"Creating container: {self.container_name}")
        logger.debug(f"Docker command: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.startup_timeout
            )

            if process.returncode != 0:
                error_msg = stderr.decode()
                raise RuntimeError(
                    f"Failed to create container '{self.container_name}':\n{error_msg}\n\n"
                    f"Troubleshooting:\n"
                    f"- Check if port {self.config.model_proxy_port} is available\n"
                    f"- Verify image exists: docker images | grep {self.config.image}\n"
                    f"- Check resource limits: memory={self.config.memory_limit}, cpus={self.config.cpu_count}"
                )

            container_id = stdout.decode().strip()
            logger.info(f"Container created successfully: {container_id[:12]}")

            return container_id

        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Container creation timed out after {self.config.startup_timeout}s.\n"
                f"This may indicate:\n"
                f"- Docker daemon is overloaded\n"
                f"- Network configuration issues\n"
                f"- Image pull in progress (check: docker images)\n"
                f"Consider increasing startup_timeout in config."
            )
    
    def _build_docker_run_cmd(self) -> list:
        """构建 docker run 命令"""
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--network", self.config.network_mode,
            "-m", self.config.memory_limit,
            f"--cpus={self.config.cpu_count}",
            "-v", f"{self._temp_dir}:{self.config.work_dir}",
            "-w", self.config.work_dir,
        ]
        
        # 添加环境变量
        env_vars = {
            "OPENAI_API_BASE": f"http://{self.config.model_proxy_host}:{self.config.model_proxy_port}/v1",
            "OPENAI_API_KEY": "verl-swe-agent",  # Placeholder
            **self.config.env_vars,
        }
        
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # 镜像和命令（保持容器运行）
        cmd.extend([self.config.image, "sleep", "infinity"])
        
        return cmd
    
    async def _init_repo(self, repo_content: Dict[str, str]) -> None:
        """初始化仓库内容
        
        Args:
            repo_content: 文件内容字典
        """
        if not self._temp_dir:
            raise RuntimeError("Temp directory not created")
        
        # 创建文件
        for filename, content in repo_content.items():
            filepath = os.path.join(self._temp_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        
        # 初始化 git 仓库
        await self._exec_local(f"cd {self._temp_dir} && git init")
        await self._exec_local(f"cd {self._temp_dir} && git config user.email 'test@verl.ai'")
        await self._exec_local(f"cd {self._temp_dir} && git config user.name 'VERL'")
        await self._exec_local(f"cd {self._temp_dir} && git add -A")
        await self._exec_local(f"cd {self._temp_dir} && git commit -m 'Initial commit'")
        
        logger.info(f"Initialized repo with {len(repo_content)} files")
    
    async def _exec_local(self, cmd: str) -> str:
        """在本地执行命令"""
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.warning(f"Local command failed: {cmd}\n{stderr.decode()}")
        return stdout.decode()
    
    async def exec(self, cmd: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """在容器内执行命令
        
        Args:
            cmd: 要执行的命令
            timeout: 超时时间（秒）
            
        Returns:
            (exit_code, stdout, stderr)
        """
        if not self.container_id:
            raise RuntimeError("Container not created")
        
        timeout = timeout or self.config.execution_timeout
        
        exec_cmd = ["docker", "exec", self.container_id, "bash", "-c", cmd]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return process.returncode, stdout.decode(), stderr.decode()
            
        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {timeout}s: {cmd[:100]}...")
            return -1, "", f"Command timed out after {timeout}s"
    
    async def run_swe_agent(
        self,
        problem_statement: str,
        instance_id: str = "test",
    ) -> Tuple[int, str]:
        """在容器内运行 SWE-Agent
        
        Args:
            problem_statement: 问题描述
            instance_id: 实例 ID
            
        Returns:
            (exit_code, output)
        """
        # 生成 SWE-Agent 配置
        config_content = self._generate_swe_agent_config(instance_id)
        
        # 写入配置文件
        config_path = os.path.join(self._temp_dir, "swe_agent_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        
        # 运行 SWE-Agent
        import shlex
        cmd = (
            f"sweagent run "
            f"--config {self.config.work_dir}/swe_agent_config.yaml "
            f"--problem_statement.text {shlex.quote(problem_statement)}"
        )
        
        logger.info(f"Running SWE-Agent: {cmd[:100]}...")
        
        exit_code, stdout, stderr = await self.exec(cmd, timeout=self.config.execution_timeout)
        
        output = stdout + stderr
        logger.info(f"SWE-Agent completed with exit code {exit_code}")
        
        return exit_code, output
    
    def _generate_swe_agent_config(self, instance_id: str) -> str:
        """生成 SWE-Agent YAML 配置"""
        builder = SWEAgentConfigBuilder(
            instance_id=instance_id,
            repo_path=self.config.work_dir,
            output_dir=f"{self.config.work_dir}/output",
            model_proxy_port=self.config.model_proxy_port,
            max_steps=100,  # Could be configurable
            execution_timeout=300,  # Could be configurable
        )

        return builder.to_yaml()
    
    async def get_patch(self) -> Optional[str]:
        """获取生成的 patch

        Returns:
            patch 内容，如果没有变更则返回 None
        """
        if not self._temp_dir:
            return None

        # Use PatchExtractor for unified logic
        extractor = PatchExtractor(
            output_dir=os.path.join(self._temp_dir, "output"),
            instance_id="",  # Not used in Docker mode pattern matching
            repo_path=self._temp_dir,
        )

        return await extractor.extract()
    
    async def cleanup(self) -> None:
        """清理容器和临时文件"""
        if self.container_id:
            try:
                # 停止容器
                process = await asyncio.create_subprocess_exec(
                    "docker", "stop", "-t", "5", self.container_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.communicate(), timeout=30)
                
                # 删除容器
                process = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", self.container_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.communicate(), timeout=30)
                
                logger.info(f"Container {self.container_id[:12]} removed")
            except Exception as e:
                logger.warning(f"Failed to cleanup container: {e}")
            finally:
                self.container_id = None
        
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Temp directory removed: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
            finally:
                self._temp_dir = None
    
    async def _build_image(self) -> None:
        """构建 Docker 镜像"""
        dockerfile_dir = os.path.dirname(os.path.abspath(__file__))
        dockerfile_path = os.path.join(dockerfile_dir, "Dockerfile")
        
        if not os.path.exists(dockerfile_path):
            raise RuntimeError(
                f"Dockerfile not found at {dockerfile_path}. "
                "Please build the image manually or create the Dockerfile."
            )
        
        logger.info(f"Building image {self.config.image}...")
        
        process = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", self.config.image, dockerfile_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=600  # 10 minutes for build
        )
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to build image: {stderr.decode()}")
        
        logger.info(f"Image {self.config.image} built successfully")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.cleanup()
        return False
