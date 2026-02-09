#!/usr/bin/env bash
# =============================================================================
# Stop Ray Cluster + Clean up Docker containers on both nodes
#
# Usage:
#   bash recipe/swe_agent/example/stop_ray_cluster.sh
# =============================================================================

set -euo pipefail

HEAD_IP=8.92.9.152
WORKER_IP=8.92.9.150
SSH_PORT=1314
SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "=== Stopping Ray cluster and cleaning up ==="

# 1. Stop Ray on both nodes
echo "[1/4] Stopping Ray on HEAD (${HEAD_IP})..."
ray stop 2>/dev/null || true

echo "[2/4] Stopping Ray on WORKER (${WORKER_IP})..."
${SSH_CMD} root@${WORKER_IP} "ray stop 2>/dev/null || true" 2>/dev/null || echo "  (worker unreachable, skipping)"

# 2. Clean swerex Docker containers on both nodes
echo "[3/4] Cleaning swerex containers on HEAD..."
CONTAINERS=$(docker ps -q --filter "ancestor=swerex-python:3.11" 2>/dev/null)
if [ -n "$CONTAINERS" ]; then
    docker stop $CONTAINERS 2>/dev/null || true
    echo "  Stopped $(echo "$CONTAINERS" | wc -w) container(s)"
else
    echo "  No containers to clean"
fi

echo "[4/4] Cleaning swerex containers on WORKER..."
${SSH_CMD} root@${WORKER_IP} '
    CONTAINERS=$(docker ps -q --filter "ancestor=swerex-python:3.11" 2>/dev/null)
    if [ -n "$CONTAINERS" ]; then
        docker stop $CONTAINERS 2>/dev/null || true
        echo "  Stopped $(echo "$CONTAINERS" | wc -w) container(s)"
    else
        echo "  No containers to clean"
    fi
' 2>/dev/null || echo "  (worker unreachable, skipping)"

# 3. Kill any residual training processes
echo "[CLEANUP] Killing residual training processes..."
pkill -f "main_ppo" 2>/dev/null || true
${SSH_CMD} root@${WORKER_IP} "pkill -f 'main_ppo' 2>/dev/null || true" 2>/dev/null || true

echo ""
echo "=== Cleanup complete ==="
echo "  Verify: ray status (should show 'No cluster')"
echo "  Verify: docker ps --filter 'ancestor=swerex-python:3.11' (should be empty)"
