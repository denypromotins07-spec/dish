#!/usr/bin/env bash
# Stage 30: Chapter 5 - File 1
# Master Launch Script - One-Click Deployment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

echo "=============================================="
echo "  CRYPTO TRADING BOT - MASTER LAUNCH"
echo "  AMD Ryzen AI 5 + Radeon GPU Optimized"
echo "=============================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    log_warn "Running as root. Consider using a regular user account."
fi

# Step 1: System Tuning for AMD Ryzen
log_info "Step 1: Configuring CPU governor for performance..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g performance 2>/dev/null || log_warn "Could not set CPU governor (permission denied or unsupported)"
else
    log_warn "cpupower not found, skipping CPU governor configuration"
fi

# Step 2: Allocate Hugepages (optional, for LMDB)
log_info "Step 2: Allocating hugepages for LMDB..."
if [[ -w /proc/sys/vm/nr_hugepages ]]; then
    echo 512 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null 2>&1 || log_warn "Could not allocate hugepages"
    log_success "Allocated 512 hugepages"
else
    log_warn "Cannot write to /proc/sys/vm/nr_hugepages, skipping"
fi

# Step 3: Build Frontend
log_info "Step 3: Building Vite frontend (production mode)..."
cd "$FRONTEND_DIR"
if command -v pnpm &> /dev/null; then
    pnpm build
elif command -v yarn &> /dev/null; then
    yarn build
else
    npm run build
fi
log_success "Frontend build complete"

# Step 4: Generate SSL Certificates (if not exists)
log_info "Step 4: Checking SSL certificates..."
CERTS_DIR="$DEPLOYMENT_DIR/security/certs"
if [[ ! -d "$CERTS_DIR" ]] || [[ ! -f "$CERTS_DIR/localhost.pem" ]]; then
    log_warn "SSL certificates not found. Running mkcert setup..."
    bash "$DEPLOYMENT_DIR/security/mkcert_setup.sh"
else
    log_success "SSL certificates already present"
fi

# Step 5: Configure Firewall Rules
log_info "Step 5: Applying firewall rules..."
if [[ -x "$DEPLOYMENT_DIR/security/firewall_rules.sh" ]]; then
    sudo bash "$DEPLOYMENT_DIR/security/firewall_rules.sh" || log_warn "Firewall configuration failed (may require manual intervention)"
fi

# Step 6: Start Docker Compose Stack
log_info "Step 6: Starting Docker Compose stack with memory limits..."
cd "$DEPLOYMENT_DIR"

# Ensure containers are stopped first
docker compose down --remove-orphans 2>/dev/null || true

# Start services
docker compose up -d --build

# Wait for services to be healthy
log_info "Waiting for services to become healthy..."
sleep 5

MAX_RETRIES=30
RETRY_COUNT=0
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    HEALTHY=$(docker compose ps --format json 2>/dev/null | jq -r 'select(.Health == "healthy") | .Service' | wc -l)
    TOTAL=$(docker compose ps --format json 2>/dev/null | jq -r '.Service' | wc -l)
    
    if [[ "$HEALTHY" -eq "$TOTAL" ]] && [[ "$TOTAL" -gt 0 ]]; then
        log_success "All $TOTAL services are healthy"
        break
    fi
    
    log_info "Waiting... ($HEALTHY/$TOTAL services healthy)"
    sleep 2
    ((RETRY_COUNT++))
done

if [[ $RETRY_COUNT -eq $MAX_RETRIES ]]; then
    log_warn "Some services may not be fully healthy yet. Check 'docker compose ps' for details."
fi

# Step 7: Open Browser
log_info "Step 7: Opening dashboard in browser..."
DASHBOARD_URL="https://localhost:3000"

if command -v xdg-open &> /dev/null; then
    xdg-open "$DASHBOARD_URL" &
elif command -v gnome-open &> /dev/null; then
    gnome-open "$DASHBOARD_URL" &
elif command -v open &> /dev/null; then
    open "$DASHBOARD_URL" &
else
    log_info "Please open $DASHBOARD_URL in your browser"
fi

# Step 8: Display Status
echo ""
echo "=============================================="
log_success "LAUNCH COMPLETE"
echo "=============================================="
echo ""
echo "Dashboard URL: ${CYAN}$DASHBOARD_URL${NC}"
echo ""
echo "Useful commands:"
echo "  - View logs:     ${CYAN}docker compose logs -f${NC}"
echo "  - Stop system:   ${CYAN}bash $DEPLOYMENT_DIR/teardown_master.sh${NC}"
echo "  - Check memory:  ${CYAN}docker stats${NC}"
echo ""
echo "Memory Limit: 14GB (enforced by docker-compose.yml)"
echo "=============================================="

# Tail logs (optional, comment out if not desired)
# docker compose logs -f
