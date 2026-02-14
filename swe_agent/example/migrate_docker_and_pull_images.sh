#!/usr/bin/env bash
# =============================================================================
# SWE-bench Docker Image Migration Script
# =============================================================================
# Purpose: 
#   1. Migrate Docker data directory from / (30GB free) to /data1 (2.7TB free)
#   2. Pull all SWE-bench eval/env/base images from 8.92.9.154 to 8.92.9.152
#   3. Copy SWE-bench dataset and code from 154
#
# Usage:
#   bash migrate_docker_and_pull_images.sh [--skip-migrate] [--skip-images] [--skip-data]
#
# Prerequisites:
#   - sshpass installed (apt-get install -y sshpass)
#   - Docker socket accessible
#   - /data1 mounted with sufficient space (>1.5TB recommended)
#
# WARNING: Docker migration will temporarily stop all containers on 152.
#          Containers with restart policy will auto-restart.
# =============================================================================

set -euo pipefail

# ================= Configuration =================
REMOTE_HOST="8.92.9.154"
REMOTE_USER="root"
REMOTE_PASS="mindrl12!@"
REMOTE_SWE_BENCH="/home/cxb/SWE-bench"
REMOTE_SWE_AGENT="/home/cxb/SWE-agent"

LOCAL_DATA_DIR="/data1/lmy/swebench"
LOCAL_DOCKER_DATA="/data1/docker"   # New Docker data root
LOCAL_IMAGE_CACHE="/data1/lmy/swebench/image_cache"  # Temp storage for image tarballs

LOG_FILE="/data1/lmy/swebench/migration_$(date +%Y%m%d_%H%M%S).log"

# Parse args
SKIP_MIGRATE=false
SKIP_IMAGES=false
SKIP_DATA=false
for arg in "$@"; do
    case $arg in
        --skip-migrate) SKIP_MIGRATE=true ;;
        --skip-images) SKIP_IMAGES=true ;;
        --skip-data) SKIP_DATA=true ;;
    esac
done

# ================= Helper functions =================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

run_on_host() {
    # Execute a command on the host machine via nsenter through Docker
    docker run --rm --privileged --net=host --pid=host \
        -v /:/host \
        ubuntu:22.04 \
        nsenter -t 1 -m -u -i -n -- bash -c "$1"
}

ssh_154() {
    sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$REMOTE_USER@$REMOTE_HOST" "$@"
}

scp_154() {
    sshpass -p "$REMOTE_PASS" scp -o StrictHostKeyChecking=no "$@"
}

# ================= Pre-flight checks =================
mkdir -p "$LOCAL_DATA_DIR" "$LOCAL_IMAGE_CACHE"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

log "=== SWE-bench Migration Script Started ==="
log "Skip migrate: $SKIP_MIGRATE, Skip images: $SKIP_IMAGES, Skip data: $SKIP_DATA"

# Check sshpass
if ! command -v sshpass &>/dev/null; then
    log "ERROR: sshpass not installed. Run: apt-get install -y sshpass"
    exit 1
fi

# Check SSH to 154
if ! ssh_154 "echo 'SSH OK'" &>/dev/null; then
    log "ERROR: Cannot SSH to $REMOTE_HOST"
    exit 1
fi
log "SSH to $REMOTE_HOST: OK"

# Check disk space
AVAIL_DATA1=$(df --output=avail /data1 | tail -1 | tr -d ' ')
log "Available space on /data1: $(( AVAIL_DATA1 / 1024 / 1024 )) GB"
if [ "$AVAIL_DATA1" -lt 1500000000 ]; then  # 1.5TB in KB
    log "WARNING: Less than 1.5TB available on /data1. May not fit all images."
fi

# =============================================================================
# Phase 1: Migrate Docker data directory to /data1
# =============================================================================
if [ "$SKIP_MIGRATE" = false ]; then
    log ""
    log "============================================"
    log "Phase 1: Migrate Docker data directory"
    log "============================================"
    
    # Check if already migrated
    CURRENT_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $NF}')
    log "Current Docker Root Dir: $CURRENT_ROOT"
    
    if [ "$CURRENT_ROOT" = "$LOCAL_DOCKER_DATA" ]; then
        log "Docker already using $LOCAL_DOCKER_DATA - skipping migration"
    else
        log "Migrating Docker data from $CURRENT_ROOT to $LOCAL_DOCKER_DATA"
        log "WARNING: This will temporarily stop all containers!"
        
        # Step 1: Stop Docker on host
        log "Stopping Docker daemon on host..."
        run_on_host "systemctl stop docker docker.socket" || true
        sleep 3
        
        # Step 2: Copy existing Docker data (if not already done)
        if [ ! -d "$LOCAL_DOCKER_DATA" ] || [ -z "$(ls -A $LOCAL_DOCKER_DATA 2>/dev/null)" ]; then
            log "Copying Docker data to $LOCAL_DOCKER_DATA (this may take a while)..."
            # Use rsync via host for reliability
            run_on_host "mkdir -p $LOCAL_DOCKER_DATA && rsync -aP /var/lib/docker/ $LOCAL_DOCKER_DATA/"
            log "Docker data copy complete"
        else
            log "$LOCAL_DOCKER_DATA already exists, syncing changes..."
            run_on_host "rsync -aP /var/lib/docker/ $LOCAL_DOCKER_DATA/"
        fi
        
        # Step 3: Update daemon.json on host
        log "Updating Docker daemon.json..."
        run_on_host "cat /etc/docker/daemon.json" > /tmp/daemon_backup.json
        
        # Create new daemon.json with data-root
        python3 -c "
import json
with open('/tmp/daemon_backup.json') as f:
    config = json.load(f)
config['data-root'] = '$LOCAL_DOCKER_DATA'
with open('/tmp/daemon_new.json', 'w') as f:
    json.dump(config, f, indent=2)
print(json.dumps(config, indent=2))
"
        
        # Write new config to host
        docker run --rm -v /:/host -v /tmp/daemon_new.json:/tmp/daemon_new.json \
            ubuntu:22.04 bash -c "cp /tmp/daemon_new.json /host/etc/docker/daemon.json"
        
        log "New daemon.json written"
        
        # Step 4: Start Docker on host
        log "Starting Docker daemon..."
        run_on_host "systemctl start docker"
        sleep 5
        
        # Verify
        NEW_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $NF}')
        log "New Docker Root Dir: $NEW_ROOT"
        
        if [ "$NEW_ROOT" = "$LOCAL_DOCKER_DATA" ]; then
            log "Docker migration successful!"
        else
            log "ERROR: Docker migration failed. Root is still $NEW_ROOT"
            log "Restoring original daemon.json..."
            docker run --rm -v /:/host -v /tmp/daemon_backup.json:/tmp/daemon_backup.json \
                ubuntu:22.04 bash -c "cp /tmp/daemon_backup.json /host/etc/docker/daemon.json"
            run_on_host "systemctl restart docker"
            exit 1
        fi
        
        # Step 5: Verify containers are back
        sleep 5
        log "Running containers after migration:"
        docker ps --format "  {{.Names}} ({{.Image}})" | tee -a "$LOG_FILE"
    fi
else
    log "Skipping Docker migration (--skip-migrate)"
fi

# =============================================================================
# Phase 2: Pull SWE-bench images from 154
# =============================================================================
if [ "$SKIP_IMAGES" = false ]; then
    log ""
    log "============================================"
    log "Phase 2: Pull SWE-bench Docker images from $REMOTE_HOST"
    log "============================================"
    
    # Get list of images on 154
    log "Fetching image list from $REMOTE_HOST..."
    IMAGE_LIST_FILE="$LOCAL_DATA_DIR/image_list_154.txt"
    ssh_154 "docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '(sweb\.|swerex)'" > "$IMAGE_LIST_FILE"
    TOTAL_IMAGES=$(wc -l < "$IMAGE_LIST_FILE")
    log "Found $TOTAL_IMAGES images on $REMOTE_HOST"
    
    # Get list of images already on 152
    LOCAL_IMAGE_FILE="$LOCAL_DATA_DIR/image_list_152.txt"
    docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '(sweb\.|swerex)' > "$LOCAL_IMAGE_FILE" 2>/dev/null || true
    
    # Find images to transfer
    NEEDED_FILE="$LOCAL_DATA_DIR/images_needed.txt"
    comm -23 <(sort "$IMAGE_LIST_FILE") <(sort "$LOCAL_IMAGE_FILE") > "$NEEDED_FILE"
    NEEDED_COUNT=$(wc -l < "$NEEDED_FILE")
    log "Images already on 152: $(wc -l < "$LOCAL_IMAGE_FILE")"
    log "Images to transfer: $NEEDED_COUNT"
    
    if [ "$NEEDED_COUNT" -eq 0 ]; then
        log "All images already present - skipping transfer"
    else
        # Transfer strategy: Use SSH pipe (docker save | ssh docker load) 
        # This avoids intermediate storage.
        # For efficiency, batch images sharing the same base layers.
        
        COUNTER=0
        FAILED=0
        
        while IFS= read -r image; do
            COUNTER=$((COUNTER + 1))
            log "[$COUNTER/$NEEDED_COUNT] Transferring: $image"
            
            # Stream directly: 154 docker save -> pipe -> 152 docker load
            # Use pv for progress if available, otherwise plain pipe
            START_TIME=$(date +%s)
            
            if ssh_154 "docker save '$image'" | docker load 2>&1; then
                ELAPSED=$(( $(date +%s) - START_TIME ))
                log "  ✓ Transferred in ${ELAPSED}s"
            else
                ELAPSED=$(( $(date +%s) - START_TIME ))
                log "  ✗ FAILED after ${ELAPSED}s"
                FAILED=$((FAILED + 1))
                # Don't abort, continue with next image
            fi
            
            # Show progress
            if [ $((COUNTER % 10)) -eq 0 ]; then
                log "Progress: $COUNTER/$NEEDED_COUNT done, $FAILED failed"
                # Check disk space
                AVAIL=$(df --output=avail /data1 | tail -1 | tr -d ' ')
                log "Remaining space: $(( AVAIL / 1024 / 1024 )) GB"
            fi
            
        done < "$NEEDED_FILE"
        
        log ""
        log "Image transfer complete: $COUNTER total, $FAILED failed"
        log "Images now on 152:"
        docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '(sweb\.|swerex)' | wc -l | xargs -I{} echo "  {} images"
    fi
else
    log "Skipping image pull (--skip-images)"
fi

# =============================================================================
# Phase 3: Copy SWE-bench code and data
# =============================================================================
if [ "$SKIP_DATA" = false ]; then
    log ""
    log "============================================"  
    log "Phase 3: Copy SWE-bench code and data from $REMOTE_HOST"
    log "============================================"
    
    # Copy SWE-bench repository
    SWE_BENCH_LOCAL="$LOCAL_DATA_DIR/SWE-bench"
    if [ -d "$SWE_BENCH_LOCAL/.git" ]; then
        log "SWE-bench repo already exists at $SWE_BENCH_LOCAL"
    else
        log "Copying SWE-bench repo..."
        mkdir -p "$SWE_BENCH_LOCAL"
        sshpass -p "$REMOTE_PASS" rsync -avz --progress \
            -e "ssh -o StrictHostKeyChecking=no" \
            "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SWE_BENCH/" \
            "$SWE_BENCH_LOCAL/" 2>&1 | tail -5
        log "SWE-bench repo copied to $SWE_BENCH_LOCAL"
    fi
    
    # Copy SWE-agent test data (contains swe-bench-lite-test.json)
    SWE_AGENT_DATA_LOCAL="$LOCAL_DATA_DIR/SWE-agent-data"
    mkdir -p "$SWE_AGENT_DATA_LOCAL"
    log "Copying SWE-agent test data..."
    sshpass -p "$REMOTE_PASS" rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SWE_AGENT/tests/test_data/data_sources/" \
        "$SWE_AGENT_DATA_LOCAL/" 2>&1 | tail -5
    log "SWE-agent test data copied"
    
    # Verify data files
    log "Data files available:"
    ls -la "$SWE_AGENT_DATA_LOCAL"/*.json 2>/dev/null | tee -a "$LOG_FILE" || log "  No .json files found"
    ls -la "$SWE_AGENT_DATA_LOCAL"/*.jsonl 2>/dev/null | tee -a "$LOG_FILE" || log "  No .jsonl files found"
else
    log "Skipping data copy (--skip-data)"
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "============================================"
log "Migration Summary"
log "============================================"
log "Docker Root: $(docker info 2>/dev/null | grep 'Docker Root Dir' | awk '{print $NF}')"
log "Docker images (swe): $(docker images --format '{{.Repository}}' | grep -cE '(sweb\.|swerex)' 2>/dev/null || echo 0)"
log "Disk usage:"
df -h /data1 | tail -1 | tee -a "$LOG_FILE"
log "SWE-bench data: $LOCAL_DATA_DIR"
log ""
log "Log file: $LOG_FILE"
log "=== Migration Complete ==="
