# SWE-Agent Docker Image for VERL
#
# 用于隔离运行 SWE-Agent 的 Docker 镜像
#
# 构建方法:
#   docker build -t verl-swe-agent:latest .
#
# 使用方法:
#   由 DockerSandbox 自动管理

FROM python:3.12-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    jq \
    && rm -rf /var/lib/apt/lists/*

# 配置 Git
RUN git config --global user.email "verl@example.com" && \
    git config --global user.name "VERL SWE-Agent" && \
    git config --global init.defaultBranch main

# 安装 SWE-Agent
RUN pip install --upgrade pip && \
    pip install sweagent

# 创建工作目录
WORKDIR /workspace

# 设置默认命令
CMD ["sleep", "infinity"]
