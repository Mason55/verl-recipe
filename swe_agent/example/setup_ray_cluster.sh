#!/usr/bin/env bash
# =============================================================================
# Setup 4-node Ray cluster for SWE-Agent VERL Training
# =============================================================================
# Topology:
#   HEAD:    8.92.9.152 (Ray head + Dashboard :8265)
#   WORKER1: 8.92.9.150 (via SSH port 1314, docker: lmy_rl)
#   WORKER2: 8.92.9.154 (via SSH port 1314, docker: lmy_rl)
#   WORKER3: 8.92.9.155 (via SSH port 1314, docker: lmy_rl)
# =============================================================================
set -euo pipefail

HEAD_IP="8.92.9.152"
WORKER_IPS=("8.92.9.150" "8.92.9.154" "8.92.9.155")
SSH_PORT=1314
RAY_PORT=6379
DASHBOARD_PORT=8265

WORK_BASE=${WORK_BASE:-/data1/lmy/workspace}
RAY_TMPDIR=$WORK_BASE/ray_tmp

log() { echo "[$(date '+%H:%M:%S')] $1"; }

ssh_worker() {
    local ip="$1"; shift
    ssh -n -p "$SSH_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes root@"$ip" "$@"
}

# =================== Step 1: Stop any existing Ray ===================
log "Stopping any existing Ray processes..."
ray stop --force 2>/dev/null || true

for ip in "${WORKER_IPS[@]}"; do
    log "  Stopping Ray on $ip..."
    ssh_worker "$ip" "ray stop --force 2>/dev/null || true" 2>&1 || true
done
sleep 2

# =================== Step 2: Start Ray head ===================
log "Starting Ray head on $HEAD_IP..."
mkdir -p "$RAY_TMPDIR"

RAY_TMPDIR="$RAY_TMPDIR" ray start \
    --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --num-gpus=8 \
    --temp-dir="$RAY_TMPDIR"

sleep 3
log "Ray head started. Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"

# =================== Step 3: Start Ray workers ===================
for ip in "${WORKER_IPS[@]}"; do
    log "Starting Ray worker on $ip..."
    ssh_worker "$ip" "
        mkdir -p $RAY_TMPDIR
        export RAY_TMPDIR=$RAY_TMPDIR
        ray start \
            --address=${HEAD_IP}:${RAY_PORT} \
            --num-gpus=8 \
            --temp-dir=$RAY_TMPDIR
    " 2>&1
    log "  Worker $ip started"
done

sleep 5

# =================== Step 4: Verify cluster ===================
log "Verifying cluster status..."
ray status

NODES=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo "?")
log ""
log "=========================================="
log "Ray cluster ready: $NODES nodes"
log "  HEAD:    $HEAD_IP"
for ip in "${WORKER_IPS[@]}"; do
    log "  WORKER:  $ip"
done
log "  Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"
log "=========================================="
