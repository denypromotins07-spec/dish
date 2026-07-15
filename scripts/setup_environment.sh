#!/bin/bash
# =============================================================================
# Environment Setup Script - Crypto Trading Bot
# Enforces Rust 1.97.0 for AMD Ryzen AI 5 optimizations
# Strict 14GB RAM constraint compliance
# =============================================================================

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Required version
REQUIRED_RUST_VERSION="1.97.0"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Step 1: Verify and Install Rust 1.97.0
# =============================================================================
log_info "Checking Rust toolchain..."

if ! command -v rustup &> /dev/null; then
    log_error "rustup is not installed. Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Check current Rust version
CURRENT_VERSION=$(rustc --version | awk '{print $2}')
log_info "Current Rust version: $CURRENT_VERSION"

if [ "$CURRENT_VERSION" != "$REQUIRED_RUST_VERSION" ]; then
    log_warning "Rust version mismatch. Required: $REQUIRED_RUST_VERSION, Found: $CURRENT_VERSION"
    log_info "Installing Rust $REQUIRED_RUST_VERSION..."
    
    rustup install "$REQUIRED_RUST_VERSION"
    rustup default "$REQUIRED_RUST_VERSION"
    
    # Verify installation
    NEW_VERSION=$(rustc --version | awk '{print $2}')
    if [ "$NEW_VERSION" != "$REQUIRED_RUST_VERSION" ]; then
        log_error "Failed to install Rust $REQUIRED_RUST_VERSION. Current: $NEW_VERSION"
        exit 1
    fi
    log_success "Rust $REQUIRED_RUST_VERSION successfully installed and set as default"
else
    log_success "Rust version is correct: $CURRENT_VERSION"
fi

# Install required components
log_info "Installing required Rust components..."
rustup component add rustfmt clippy rust-src llvm-tools --toolchain "$REQUIRED_RUST_VERSION"
rustup target add x86_64-unknown-linux-gnu --toolchain "$REQUIRED_RUST_VERSION"
rustup target add x86_64-unknown-linux-musl --toolchain "$REQUIRED_RUST_VERSION"

log_success "All Rust components installed"

# =============================================================================
# Step 2: Configure System for AMD Ryzen AI 5
# =============================================================================
log_info "Configuring system for AMD Ryzen AI 5..."

# Set CPU governor to performance (requires sudo)
if command -v cpufreq-set &> /dev/null; then
    log_info "Setting CPU governor to performance mode..."
    sudo cpufreq-set -g performance 2>/dev/null || log_warning "Could not set CPU governor (may require manual configuration)"
else
    log_warning "cpufreq-set not available. CPU governor must be set manually."
fi

# Configure hugepages for reduced TLB misses (optional, requires sudo)
log_info "Configuring hugepages..."
if [ -w /proc/sys/vm/nr_hugepages ]; then
    echo 2048 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null
    log_success "Hugepages configured: 2048 pages (~4GB)"
else
    log_warning "Cannot configure hugepages without sudo. Add to /etc/sysctl.conf: vm.nr_hugepages=2048"
fi

# =============================================================================
# Step 3: Memory Limit Configuration
# =============================================================================
log_info "Configuring memory limits for 14GB constraint..."

export MALLOC_ARENA_MAX=2
export RUST_MIN_STACK=8388608
export MIMALLOC_PAGE_SIZE=2097152
export MIMALLOC_ARENA_PER_THREAD=1

# Add to .bashrc for persistence
if ! grep -q "MALLOC_ARENA_MAX" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'

# Crypto Trading Bot Memory Limits (14GB constraint)
export MALLOC_ARENA_MAX=2
export RUST_MIN_STACK=8388608
export MIMALLOC_PAGE_SIZE=2097152
export MIMALLOC_ARENA_PER_THREAD=1
EOF
    log_success "Memory limits added to ~/.bashrc"
fi

# =============================================================================
# Step 4: Python Environment Setup
# =============================================================================
log_info "Setting up Python environment..."

PYTHON_VERSION="3.11"

if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    exit 1
fi

INSTALLED_PYTHON=$(python3 --version | awk '{print $2}')
log_info "Python version: $INSTALLED_PYTHON"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    log_info "Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install Python dependencies
log_info "Installing Python dependencies..."
pip install --upgrade pip
pip install polars duckdb pyarrow numpy pandas
pip install maturin pytest mypy black
pip install ray[tune]  # For distributed analytics

log_success "Python environment configured"

# =============================================================================
# Step 5: Build PyO3 Bindings
# =============================================================================
log_info "Building PyO3 bindings with Rust $REQUIRED_RUST_VERSION..."

if [ -d "crates/python_bindings" ]; then
    cd crates/python_bindings
    maturin develop --release
    cd ../..
    log_success "PyO3 bindings built successfully"
else
    log_warning "PyO3 bindings directory not found, skipping..."
fi

# =============================================================================
# Step 6: Pre-build Release Binary with AMD Optimizations
# =============================================================================
log_info "Building release binary with AMD Ryzen optimizations..."

export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin -C codegen-units=1"

cargo build --release --bin crypto_bot

if [ -f "target/release/crypto_bot" ]; then
    BINARY_SIZE=$(ls -lh target/release/crypto_bot | awk '{print $5}')
    log_success "Release binary built: target/release/crypto_bot ($BINARY_SIZE)"
else
    log_error "Failed to build release binary"
    exit 1
fi

# =============================================================================
# Step 7: Verification
# =============================================================================
log_info "Running verification checks..."

# Verify Rust version
FINAL_RUST_VERSION=$(rustc --version)
log_info "Rust: $FINAL_RUST_VERSION"

# Verify binary exists and is executable
if [ -x "target/release/crypto_bot" ]; then
    log_success "Binary is executable"
else
    log_error "Binary is not executable"
    exit 1
fi

# Check memory configuration
log_info "Memory configuration:"
echo "  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
echo "  RUST_MIN_STACK=$RUST_MIN_STACK"

log_success "=========================================="
log_success "Environment setup complete!"
log_success "Rust $REQUIRED_RUST_VERSION is active"
log_success "Ready for AMD Ryzen AI 5 deployment"
log_success "=========================================="

# Print next steps
echo ""
log_info "Next steps:"
echo "  1. Run './scripts/run_bot.sh' to start the trading bot"
echo "  2. Monitor memory usage: 'watch -n 1 free -h'"
echo "  3. Check logs: 'tail -f logs/crypto_bot.log'"
