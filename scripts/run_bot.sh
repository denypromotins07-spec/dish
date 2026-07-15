#!/bin/bash
# Master Deployment Script for AMD Ryzen AI Environment
# Sets CPU governor, allocates hugepages, sets thread priorities, and launches all components.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY_NAME="crypto-bot"
CONFIG_FILE="${PROJECT_ROOT}/core_config.toml"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${PROJECT_ROOT}/pids"

# Memory configuration (strict 14GB limit)
TOTAL_RAM_GB=14
HUGEPAGES_COUNT=2048  # 4MB pages = 8GB reserved

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Crypto Trading Bot Deployment Script${NC}"
echo -e "${BLUE}  Target: AMD Ryzen AI + Radeon GPU${NC}"
echo -e "${BLUE}  RAM Limit: ${TOTAL_RAM_GB}GB${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root (needed for some operations)
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_warning "Some operations may require root privileges. Run with sudo if needed."
    fi
}

# Create required directories
setup_directories() {
    echo ""
    echo -e "${BLUE}Setting up directories...${NC}"
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$PID_DIR"
    mkdir -p "${PROJECT_ROOT}/data/lmdb"
    mkdir -p "${PROJECT_ROOT}/data/duckdb"
    
    print_status "Directories created"
}

# Set CPU governor to performance mode
set_cpu_governor() {
    echo ""
    echo -e "${BLUE}Configuring CPU governor...${NC}"
    
    # Check if cpufreq is available
    if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -w "$cpu" ]; then
                echo "performance" | sudo tee "$cpu" > /dev/null 2>&1 || true
            fi
        done
        
        # Verify
        GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
        if [ "$GOVERNOR" = "performance" ]; then
            print_status "CPU governor set to performance mode"
        else
            print_warning "Could not set CPU governor (current: $GOVERNOR)"
        fi
    else
        print_warning "cpufreq not available, skipping CPU governor setup"
    fi
}

# Configure hugepages for low-latency memory
setup_hugepages() {
    echo ""
    echo -e "${BLUE}Configuring hugepages...${NC}"
    
    # Try to set hugepages (requires root)
    if command -v sysctl &> /dev/null; then
        CURRENT_HUGEPAGES=$(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo "0")
        
        if [ "$CURRENT_HUGEPAGES" -lt "$HUGEPAGES_COUNT" ]; then
            print_warning "Requesting $HUGEPAGES_COUNT hugepages (current: $CURRENT_HUGEPAGES)"
            sudo sysctl -w vm.nr_hugepages=$HUGEPAGES_COUNT 2>/dev/null || {
                print_warning "Failed to set hugepages (may require reboot or manual config)"
                echo "To set manually, add to /etc/sysctl.conf: vm.nr_hugepages=$HUGEPAGES_COUNT"
            }
        else
            print_status "Hugepages already configured ($CURRENT_HUGEPAGES pages)"
        fi
    else
        print_warning "sysctl not available, skipping hugepages setup"
    fi
    
    # Set memlock limits for current session
    ulimit -l unlimited 2>/dev/null || print_warning "Could not set memlock limit"
}

# Set thread priorities using chrt
set_thread_priorities() {
    echo ""
    echo -e "${BLUE}Configuring thread priorities...${NC}"
    
    # Check if RT scheduling is available
    if command -v chrt &> /dev/null; then
        # Get current max RT priority
        MAX_RT_PRIO=$(chrt -m 2>/dev/null | grep -oP 'maximum .* priority: \K\d+' || echo "0")
        
        if [ "$MAX_RT_PRIO" -gt 0 ]; then
            print_status "Real-time scheduling available (max priority: $MAX_RT_PRIO)"
            
            # Store priority info for later use
            export BOT_RT_PRIORITY=$((MAX_RT_PRIO / 2))
            export BOT_SCHED_POLICY="SCHED_FIFO"
        else
            print_warning "RT scheduling not available or restricted"
            export BOT_RT_PRIORITY=0
            export BOT_SCHED_POLICY="SCHED_OTHER"
        fi
    else
        print_warning "chrt not available, using default priorities"
    fi
}

# Validate configuration before starting
validate_config() {
    echo ""
    echo -e "${BLUE}Validating configuration...${NC}"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Run Python config validator
    if command -v python3 &> /dev/null; then
        python3 "${PROJECT_ROOT}/python/cli/config_validator.py" "$CONFIG_FILE" || {
            print_error "Configuration validation failed"
            exit 1
        }
        print_status "Configuration validated"
    else
        print_warning "Python3 not available, skipping config validation"
    fi
}

# Run pre-flight checks
run_preflight() {
    echo ""
    echo -e "${BLUE}Running pre-flight checks...${NC}"
    
    # Check Rust binary
    if [ -f "${PROJECT_ROOT}/target/release/${BINARY_NAME}" ]; then
        print_status "Rust binary found"
    else
        print_warning "Release binary not found, will attempt to build"
    fi
    
    # Check Python dependencies
    python3 -c "import psutil; import numpy" 2>/dev/null || {
        print_warning "Some Python dependencies missing. Install with: pip install psutil numpy duckdb pyyaml"
    }
    
    # Check Redis (optional)
    if command -v redis-cli &> /dev/null; then
        redis-cli ping > /dev/null 2>&1 && print_status "Redis available" || print_warning "Redis not running"
    else
        print_warning "Redis not installed"
    fi
    
    # Check network connectivity
    if command -v curl &> /dev/null; then
        curl -s --connect-timeout 5 https://api.binance.com/api/v3/time > /dev/null && \
            print_status "Exchange API reachable" || \
            print_warning "Cannot reach exchange API"
    fi
}

# Build the Rust binary if needed
build_binary() {
    echo ""
    echo -e "${BLUE}Building Rust binary...${NC}"
    
    cd "$PROJECT_ROOT"
    
    if command -v cargo &> /dev/null; then
        # Release build with optimizations
        cargo build --release --bin "$BINARY_NAME" || {
            print_error "Build failed"
            exit 1
        }
        print_status "Binary built successfully"
    else
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi
}

# Start the Python supervisor
start_python_supervisor() {
    echo ""
    echo -e "${BLUE}Starting Python supervisor...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Start in background
    nohup python3 "${PROJECT_ROOT}/python/orchestrator/python_supervisor.py" \
        > "${LOG_DIR}/python_supervisor.log" 2>&1 &
    
    SUPERVISOR_PID=$!
    echo $SUPERVISOR_PID > "${PID_DIR}/python_supervisor.pid"
    print_status "Python supervisor started (PID: $SUPERVISOR_PID)"
}

# Start the Rust bot
start_rust_bot() {
    echo ""
    echo -e "${BLUE}Starting Rust trading bot...${NC}"
    
    cd "$PROJECT_ROOT"
    
    BINARY_PATH="${PROJECT_ROOT}/target/release/${BINARY_NAME}"
    
    # Build launch command with appropriate priorities
    LAUNCH_CMD=""
    
    if [ -n "$BOT_RT_PRIORITY" ] && [ "$BOT_RT_PRIORITY" -gt 0 ]; then
        LAUNCH_CMD="chrt -f $BOT_RT_PRIORITY"
    fi
    
    # Add nice level for non-RT systems
    LAUNCH_CMD="$LAUNCH_CMD nice -n -10"
    
    # Launch with environment variables
    export RUST_LOG="info"
    export EXCHANGE_API_KEY="${EXCHANGE_API_KEY:-}"
    export EXCHANGE_API_SECRET="${EXCHANGE_API_SECRET:-}"
    
    nohup $LAUNCH_CMD "$BINARY_PATH" \
        --config "$CONFIG_FILE" \
        --log-level info \
        start \
        > "${LOG_DIR}/bot.log" 2>&1 &
    
    BOT_PID=$!
    echo $BOT_PID > "${PID_DIR}/bot.pid"
    print_status "Trading bot started (PID: $BOT_PID)"
}

# Run warmup pipeline
run_warmup() {
    echo ""
    echo -e "${BLUE}Running warmup pipeline...${NC}"
    
    cd "$PROJECT_ROOT"
    
    python3 "${PROJECT_ROOT}/python/boot/warmup_pipeline.py" \
        --symbols BTC-USD ETH-USD SOL-USD \
        --samples 1000 \
        --workers 4 || {
        print_warning "Warmup pipeline had issues"
    }
    
    print_status "Warmup complete"
}

# Graceful shutdown function
shutdown() {
    echo ""
    echo -e "${YELLOW}Initiating graceful shutdown...${NC}"
    
    # Stop Rust bot first
    if [ -f "${PID_DIR}/bot.pid" ]; then
        BOT_PID=$(cat "${PID_DIR}/bot.pid")
        if kill -0 "$BOT_PID" 2>/dev/null; then
            echo "Stopping bot (PID: $BOT_PID)..."
            kill -TERM "$BOT_PID" || true
            sleep 2
            kill -9 "$BOT_PID" 2>/dev/null || true
        fi
        rm -f "${PID_DIR}/bot.pid"
    fi
    
    # Stop Python supervisor
    if [ -f "${PID_DIR}/python_supervisor.pid" ]; then
        SUPERVISOR_PID=$(cat "${PID_DIR}/python_supervisor.pid")
        if kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
            echo "Stopping supervisor (PID: $SUPERVISOR_PID)..."
            kill -TERM "$SUPERVISOR_PID" || true
            sleep 2
            kill -9 "$SUPERVISOR_PID" 2>/dev/null || true
        fi
        rm -f "${PID_DIR}/python_supervisor.pid"
    fi
    
    print_status "Shutdown complete"
}

# Show status
show_status() {
    echo ""
    echo -e "${BLUE}System Status:${NC}"
    
    # Check bot process
    if [ -f "${PID_DIR}/bot.pid" ]; then
        BOT_PID=$(cat "${PID_DIR}/bot.pid")
        if kill -0 "$BOT_PID" 2>/dev/null; then
            echo -e "  Bot: ${GREEN}Running${NC} (PID: $BOT_PID)"
        else
            echo -e "  Bot: ${RED}Not Running${NC}"
        fi
    else
        echo -e "  Bot: ${YELLOW}No PID file${NC}"
    fi
    
    # Check supervisor
    if [ -f "${PID_DIR}/python_supervisor.pid" ]; then
        SUPERVISOR_PID=$(cat "${PID_DIR}/python_supervisor.pid")
        if kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
            echo -e "  Supervisor: ${GREEN}Running${NC} (PID: $SUPERVISOR_PID)"
        else
            echo -e "  Supervisor: ${RED}Not Running${NC}"
        fi
    else
        echo -e "  Supervisor: ${YELLOW}No PID file${NC}"
    fi
    
    # Memory usage
    echo ""
    echo "Memory Usage:"
    if command -v free &> /dev/null; then
        free -h | grep -E "^Mem:"
    fi
    
    # Hugepages
    if [ -f /proc/sys/vm/nr_hugepages ]; then
        echo "Hugepages: $(cat /proc/sys/vm/nr_hugepages) pages"
    fi
}

# Main execution
main() {
    case "${1:-start}" in
        start)
            check_root
            setup_directories
            set_cpu_governor
            setup_hugepages
            set_thread_priorities
            validate_config
            run_preflight
            
            # Build if binary doesn't exist
            if [ ! -f "${PROJECT_ROOT}/target/release/${BINARY_NAME}" ]; then
                build_binary
            fi
            
            # Run warmup
            run_warmup
            
            # Start services
            start_python_supervisor
            start_rust_bot
            
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}  Deployment Complete!${NC}"
            echo -e "${GREEN}========================================${NC}"
            show_status
            ;;
        stop)
            shutdown
            ;;
        restart)
            shutdown
            sleep 2
            main start
            ;;
        status)
            show_status
            ;;
        build)
            build_binary
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|build}"
            exit 1
            ;;
    esac
}

# Handle script arguments
main "$@"
