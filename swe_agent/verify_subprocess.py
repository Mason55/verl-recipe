#!/usr/bin/env python3
"""
SWE-Agent Subprocess 模式验证脚本

验证流程：
1. 启动 ModelProxy
2. 创建测试仓库
3. 启动 SWE-Agent（通过 subprocess）
4. 拦截模型请求并返回 mock 响应
5. 验证 patch 生成

使用方法:
    python3.12 recipe/swe_agent/verify_subprocess.py
"""

import asyncio
import logging
import os
import shutil
import sys
import tempfile

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_subprocess")

# 添加项目路径 - 支持直接运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # verl/
sys.path.insert(0, VERL_ROOT)


async def create_test_repo(repo_path: str) -> None:
    """创建测试仓库"""
    os.makedirs(repo_path, exist_ok=True)
    
    # 创建测试文件
    with open(os.path.join(repo_path, "1.txt"), "w") as f:
        f.write("Hello World\n")
    
    # 初始化 git
    cmds = [
        f"cd {repo_path} && git init",
        f"cd {repo_path} && git config user.email 'test@verl.ai'",
        f"cd {repo_path} && git config user.name 'VERL'",
        f"cd {repo_path} && git add -A",
        f"cd {repo_path} && git commit -m 'Initial commit'",
    ]
    
    for cmd in cmds:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    
    logger.info(f"Test repo created at {repo_path}")


async def mock_model_responder(proxy, max_calls: int = 10) -> int:
    """Mock 模型响应器
    
    拦截 SWE-Agent 的请求并返回简单的响应
    """
    call_count = 0
    
    while call_count < max_calls:
        try:
            # 等待请求，超时 60 秒
            request = await asyncio.wait_for(
                proxy.get_request(),
                timeout=60.0
            )
            
            call_count += 1
            logger.info(f"[Call {call_count}] Received request: {len(request.messages)} messages")
            
            # 生成简单的响应
            if call_count == 1:
                # 第一次调用：执行重命名命令
                response = """I'll rename 1.txt to 2.txt using the mv command.

<function_call>
{"name": "bash", "arguments": {"command": "mv 1.txt 2.txt"}}
</function_call>"""
            elif call_count == 2:
                # 第二次调用：提交更改
                response = """The file has been renamed. Let me commit the change.

<function_call>
{"name": "bash", "arguments": {"command": "git add -A && git diff --cached"}}
</function_call>"""
            else:
                # 后续调用：提交完成
                response = """The task is complete. The file 1.txt has been successfully renamed to 2.txt.

<function_call>
{"name": "submit", "arguments": {}}
</function_call>"""
            
            # 发送响应
            await proxy.send_response(response, request=request)
            logger.info(f"[Call {call_count}] Sent response")
            
            # 如果是 submit，结束循环
            if "submit" in response.lower():
                logger.info("Submit detected, ending mock responder")
                break
                
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for request")
            break
        except Exception as e:
            logger.error(f"Error in mock responder: {e}")
            break
    
    return call_count


async def run_verification():
    """运行验证"""
    # 直接从本地模块导入
    from model_proxy import ModelProxy
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="swe_verify_")
    repo_path = os.path.join(temp_dir, "repo")
    output_dir = os.path.join(temp_dir, "output")
    
    logger.info(f"Working directory: {temp_dir}")
    
    # 创建测试仓库
    await create_test_repo(repo_path)
    
    # 启动 ModelProxy
    proxy = ModelProxy(port=8082)  # 使用不同端口避免冲突
    await proxy.start_server()
    logger.info("ModelProxy started on port 8082")
    
    try:
        # 生成 SWE-Agent 配置
        import yaml
        
        config = {
            "output_dir": output_dir,
            "env": {
                "repo": {
                    "path": repo_path,
                    "type": "local",
                },
                "deployment": {"type": "local"},
                "name": "verify-test",
            },
            "problem_statement": {
                "type": "text",
                "text": "",
                "id": "verify_test",
            },
            "agent": {
                "model": {
                    "name": "openai/test-model",
                    "api_base": "http://127.0.0.1:8082/v1/",
                    "api_key": "test-key",
                    "per_instance_cost_limit": 0,
                    "per_instance_call_limit": 20,
                    "temperature": 0.0,
                },
                "tools": {
                    "execution_timeout": 60,
                    "bundles": [
                        {"path": "tools/registry"},
                        {"path": "tools/edit_anthropic"},
                    ],
                    "enable_bash_tool": True,
                    "parse_function": {"type": "function_calling"},
                },
            },
        }
        
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # 启动 SWE-Agent
        import shlex
        problem = "rename 1.txt to 2.txt"
        
        # 使用 python -m sweagent 而不是直接调用 sweagent
        cmd = [
            sys.executable, "-m", "sweagent", "run",
            "--config", config_path,
            "--problem_statement.text", problem,
        ]
        
        env = os.environ.copy()
        env["OPENAI_API_BASE"] = "http://127.0.0.1:8082/v1"
        env["OPENAI_API_KEY"] = "test-key"
        
        logger.info(f"Starting SWE-Agent: {' '.join(cmd[:4])}...")
        
        # 并行运行 SWE-Agent 和 mock 响应器
        agent_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        
        # 运行 mock 响应器
        responder_task = asyncio.create_task(mock_model_responder(proxy, max_calls=10))
        
        # 等待 SWE-Agent 完成（超时 5 分钟）
        try:
            stdout, stderr = await asyncio.wait_for(
                agent_proc.communicate(),
                timeout=300.0
            )
            logger.info(f"SWE-Agent completed with exit code {agent_proc.returncode}")
            
            if agent_proc.returncode != 0:
                logger.warning(f"SWE-Agent stderr: {stderr.decode()[:500]}")
                
        except asyncio.TimeoutError:
            logger.error("SWE-Agent timed out")
            agent_proc.kill()
            await agent_proc.wait()
        
        # 等待响应器完成
        call_count = await responder_task
        logger.info(f"Total model calls: {call_count}")
        
        # 检查结果
        # 检查文件是否被重命名
        file_1_exists = os.path.exists(os.path.join(repo_path, "1.txt"))
        file_2_exists = os.path.exists(os.path.join(repo_path, "2.txt"))
        
        logger.info(f"Verification results:")
        logger.info(f"  - 1.txt exists: {file_1_exists}")
        logger.info(f"  - 2.txt exists: {file_2_exists}")
        logger.info(f"  - Model calls: {call_count}")
        
        # 检查 patch
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "HEAD",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        patch = stdout.decode().strip()
        
        if patch:
            logger.info(f"  - Patch generated: {len(patch)} chars")
            print("\n=== Generated Patch ===")
            print(patch[:500])
        else:
            logger.info("  - No patch (changes may be committed)")
        
        # 判断成功
        success = call_count > 0
        
        if success:
            print("\n✅ Verification PASSED: SWE-Agent interacted with ModelProxy")
        else:
            print("\n❌ Verification FAILED: No model calls detected")
        
        return success
        
    finally:
        # 清理
        await proxy.stop_server()
        logger.info("ModelProxy stopped")
        
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("SWE-Agent Subprocess Mode Verification")
    print("=" * 60)
    
    success = asyncio.run(run_verification())
    sys.exit(0 if success else 1)
