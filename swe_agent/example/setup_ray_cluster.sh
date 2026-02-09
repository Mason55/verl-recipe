#!/usr/bin/env bash
# =============================================================================
# Ray Cluster Setup for 2-Node SWE-Agent Training
#
# Run this script on the HEAD node (8.92.9.152) to:
#   1. Start Ray head process locally
#   2. Start Ray worker process on 8.92.9.150 via SSH
#   3. Wait until both nodes are ready
#
# Prerequisites:
#   - SSH key-based auth between 152 and 150 (port 1314)
#   - Ray installed on both nodes
#   - Same code/model/data paths on both nodes
#
# Usage:
#   bash recipe/swe_agent/example/setup_ray_cluster.sh        # start cluster
#   bash recipe/swe_agent/example/setup_ray_cluster.sh stop   # stop cluster
# =============================================================================

set -euo pipefail

# ================= Cluster Config =================
HEAD_IP=8.92.9.152
WORKER_IP=8.92.9.150
SSH_PORT=1314
RAY_PORT=6379
GPUS_PER_NODE=8
EXPECTED_NODES=2

# SSH shorthand
SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# ================= Stop mode =================
if [ "${1:-}" = "stop" ]; then
    echo "=== Stopping Ray cluster ==="
    echo "[HEAD] Stopping Ray on ${HEAD_IP}..."
    ray stop 2>/dev/null || true

    echo "[WORKER] Stopping Ray on ${WORKER_IP}..."
    ${SSH_CMD} root@${WORKER_IP} "ray stop 2>/dev/null || true"

    # Clean up any residual swerex containers on both nodes
    echo "[HEAD] Cleaning swerex containers..."
    docker stop $(docker ps -q --filter "ancestor=swerex-python:3.11") 2>/dev/null || true

    echo "[WORKER] Cleaning swerex containers..."
    ${SSH_CMD} root@${WORKER_IP} \
        'docker stop $(docker ps -q --filter "ancestor=swerex-python:3.11") 2>/dev/null || true'

    echo "=== Ray cluster stopped ==="
    exit 0
fi

# ================= Start mode =================
echo "=========================================="
echo "  Setting up 2-Node Ray Cluster"
echo "  HEAD:   ${HEAD_IP} (${GPUS_PER_NODE} GPUs)"
echo "  WORKER: ${WORKER_IP} (${GPUS_PER_NODE} GPUs)"
echo "=========================================="

# Check SSH connectivity
echo "[CHECK] Testing SSH to worker node..."
${SSH_CMD} root@${WORKER_IP} "echo 'SSH OK'" || {
    echo "[ERROR] Cannot SSH to ${WORKER_IP}:${SSH_PORT}"
    echo "  Fix: ssh-copy-id -p ${SSH_PORT} root@${WORKER_IP}"
    exit 1
}

# Stop any existing Ray instances first
echo "[CLEANUP] Stopping existing Ray instances..."
ray stop 2>/dev/null || true
${SSH_CMD} root@${WORKER_IP} "ray stop 2>/dev/null || true"
sleep 2

# Start Ray head node
echo "[HEAD] Starting Ray head on ${HEAD_IP}:${RAY_PORT}..."
ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir=/data1/lmy/workspace/ray_tmp

echo "[HEAD] Ray head started. Waiting 5s for stabilization..."
sleep 5

# Start Ray worker node via SSH
echo "[WORKER] Starting Ray worker on ${WORKER_IP}..."
${SSH_CMD} root@${WORKER_IP} \
    "ray start --address='${HEAD_IP}:${RAY_PORT}' \
     --num-gpus=${GPUS_PER_NODE} \
     --temp-dir=/data1/lmy/workspace/ray_tmp"

echo "[WORKER] Ray worker started. Waiting for cluster formation..."

# Wait for both nodes to be ready
MAX_WAIT=120
WAITED=0
while true; do
    NODE_COUNT=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
    if [ "$NODE_COUNT" -ge "$EXPECTED_NODES" ]; then
        echo ""
        echo "=========================================="
        echo "  Ray cluster ready! ${NODE_COUNT} nodes connected"
        echo "  Dashboard: http://${HEAD_IP}:8265"
        echo "=========================================="
        ray status
        break
    fi

    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo ""
        echo "[ERROR] Timeout waiting for ${EXPECTED_NODES} nodes (got ${NODE_COUNT})"
        echo "  Check: ray status"
        echo "  Check worker: ${SSH_CMD} root@${WORKER_IP} 'ray status'"
        exit 1
    fi

    printf "\r  Waiting for nodes... (%d/%d, %ds elapsed)" "$NODE_COUNT" "$EXPECTED_NODES" "$WAITED"
    sleep 5
done
