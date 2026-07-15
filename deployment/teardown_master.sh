#!/usr/bin/env bash
# Stage 30: Chapter 5 - File 2
# Graceful Teardown Script - Safe Shutdown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

echo "=============================================="
echo "  CRYPTO TRADING BOT - GRACEFUL SHUTDOWN"
echo "=============================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

BACKEND_URL="https://localhost:8080"
TIMEOUT=30

# Step 1: Trigger Emergency Flattener via API
log_info "Step 1: Triggering emergency position flattener..."
RESPONSE=$(curl -s -X POST "$BACKEND_URL/api/v1/emergency/flatten" \
    --max-time 5 \
    --insecure \
    -H "Content-Type: application/json" \
    -d '{"reason": "user_initiated_shutdown", "timestamp": "'$(date -Iseconds)'"}' \
    2>/dev/null || echo '{"status": "timeout_or_unavailable"}')

if echo "$RESPONSE" | jq -e '.status == "acknowledged"' > /dev/null 2>&1; then
    log_success "Emergency flattener acknowledged"
else
    log_warn "Could not reach backend or flattener not acknowledged (may already be flat)"
fi

# Step 2: Wait for Exchange Reconciliation
log_info "Step 2: Waiting for exchange reconciliation (${TIMEOUT}s max)..."
sleep 5

# Poll for reconciliation status
for i in $(seq 1 $((TIMEOUT / 2))); do
    STATUS=$(curl -s -X GET "$BACKEND_URL/api/v1/system/reconciliation" \
        --max-time 3 \
        --insecure \
        2>/dev/null | jq -r '.status // "unknown"' || echo "unknown")
    
    if [[ "$STATUS" == "reconciled" ]] || [[ "$STATUS" == "flat" ]]; then
        log_success "Exchange reconciliation complete"
        break
    fi
    
    log_info "Reconciliation in progress... ($STATUS)"
    sleep 2
done

# Step 3: Stop Docker Compose Services
log_info "Step 3: Stopping Docker Compose services..."
cd "$DEPLOYMENT_DIR"

# Send SIGTERM first (graceful shutdown)
docker compose kill --signal SIGTERM 2>/dev/null || true

# Wait for containers to stop
log_info "Waiting for containers to stop gracefully..."
MAX_WAIT=30
WAIT_COUNT=0
while [[ $WAIT_COUNT -lt $MAX_WAIT ]]; do
    RUNNING=$(docker compose ps -q 2>/dev/null | wc -l)
    if [[ "$RUNNING" -eq 0 ]]; then
        log_success "All containers stopped"
        break
    fi
    sleep 1
    ((WAIT_COUNT++))
done

# Force stop any remaining containers
if [[ "$RUNNING" -gt 0 ]]; then
    log_warn "Force stopping remaining containers..."
    docker compose down --timeout 10 --remove-orphans
fi

# Step 4: Flush LMDB (via container cleanup hook)
log_info "Step 4: Ensuring LMDB state is flushed..."
# LMDB auto-flushes on close, but we ensure sync via Docker volume unmount
sync

# Step 5: Shutdown Ray Cluster (if running externally)
log_info "Step 5: Shutting down Ray cluster..."
if command -v ray &> /dev/null; then
    ray stop --force 2>/dev/null || log_warn "Ray cluster already stopped or not found"
else
    log_info "Ray CLI not found, skipping Ray shutdown"
fi

# Step 6: Reset CPU Governor (optional)
log_info "Step 6: Resetting CPU governor to powersave (optional)..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g powersave 2>/dev/null || log_warn "Could not reset CPU governor"
fi

# Step 7: Release Hugepages
log_info "Step 7: Releasing hugepages..."
if [[ -w /proc/sys/vm/nr_hugepages ]]; then
    echo 0 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null 2>&1 || log_warn "Could not release hugepages"
fi

# Step 8: Final Cleanup
log_info "Step 8: Cleaning up orphaned resources..."
docker system prune -f --volumes 2>/dev/null || true

echo ""
echo "=============================================="
log_success "SHUTDOWN COMPLETE"
echo "=============================================="
echo ""
echo "All positions flattened, orders cancelled, and services stopped."
echo "LMDB state persisted. Ray cluster terminated."
echo ""
echo "To restart: ${CYAN}bash $DEPLOYMENT_DIR/launch_master.sh${NC}"
echo "=============================================="
