# Ultra-Low-Latency Crypto Trading Bot
## Architecture & User Guide

### 🏗️ System Architecture Overview

This institutional-grade crypto trading bot is engineered for **sub-microsecond latency** on consumer AMD hardware (Ryzen AI 5, Radeon GPU, 16GB RAM). The system strictly adheres to a **14GB total RAM ceiling** while processing 10,000+ ticks/second.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ WebGL Canvas │  │ Zustand Store│  │ WebSocket Binary     │   │
│  │ Renderer     │  │ (HF/LF Split)│  │ Client (Zero-Copy)   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ wss:// (Protobuf)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              RUST CORE ENGINE (Axum + Tokio)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Order Book   │  │ Risk Manager │  │ Exchange Gateway     │   │
│  │ Aggregator   │  │ (Pre-Trade)  │  │ (Binance/Bybit/OKX)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ LMDB State   │  │ Emergency    │  │ Smart Order Router   │   │
│  │ Persistence  │  │ Flattener    │  │ (SOR)                │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ IPC (Shared Memory)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           PYTHON LAYER (Ray + Nautilus Trader)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ ML Inference │  │ Strategy     │  │ Backtest Engine      │   │
│  │ (ONNX)       │  │ Orchestrator │  │ (Parquet UI Reader)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 📋 Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Installation & Deployment](#installation--deployment)
3. [System Configuration](#system-configuration)
4. [Operating the Dashboard](#operating-the-dashboard)
5. [Emergency Procedures](#emergency-procedures)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

---

### 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | AMD Ryzen 5 | AMD Ryzen AI 5 / Ryzen 9 |
| GPU | AMD Radeon RX 6600 | AMD Radeon RX 7900 XT |
| RAM | 16GB | 32GB DDR5 |
| Storage | 512GB NVMe SSD | 1TB NVMe Gen4 SSD |
| Network | 100 Mbps | 1 Gbps Fiber |

**Critical:** The system is optimized for AMD hardware with ROCm support. Intel/NVIDIA configurations require Dockerfile modifications.

---

### 🚀 Installation & Deployment

#### Prerequisites

```bash
# Install Docker & Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install mkcert for local SSL
sudo apt install libnss3-tools
wget https://github.com/FiloSottile/mkcert/releases/latest/download/mkcert-linux-amd64
chmod +x mkcert-linux-amd64 && sudo mv mkcert-linux-amd64 /usr/local/bin/mkcert

# Install Node.js (v18+) and Rust (v1.75+)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### One-Click Launch

```bash
cd /path/to/crypto-bot/deployment
chmod +x launch_master.sh
./launch_master.sh
```

This script will:
1. Set CPU governor to `performance` mode
2. Allocate 512 hugepages for LMDB
3. Build the Vite frontend (production)
4. Generate SSL certificates
5. Configure firewall rules
6. Start all Docker containers with memory limits
7. Open the dashboard at `https://localhost:3000`

#### Graceful Shutdown

```bash
./teardown_master.sh
```

This will:
1. Trigger the emergency flattener API
2. Wait for exchange reconciliation
3. Stop all containers gracefully (SIGTERM)
4. Flush LMDB state
5. Shutdown Ray cluster
6. Reset CPU governor to `powersave`
7. Release hugepages

---

### ⚙️ System Configuration

#### Memory Limits (docker-compose.yml)

```yaml
services:
  rust-core:
    mem_limit: 4g
    cpuset: "0-3"
  
  python-ray:
    mem_limit: 8g
    cpuset: "4-7"
  
  frontend:
    mem_limit: 1g
    cpuset: "8-9"
```

**Total Reserved:** 13GB (leaves 2GB buffer for OS)

#### Environment Variables (.env)

```bash
# Exchange API Keys (encrypted via env_encryptor.py)
ENCRYPTED_BINANCE_KEY=<encrypted_blob>
ENCRYPTED_BYBIT_KEY=<encrypted_blob>

# Risk Parameters
MAX_LEVERAGE=5
MAX_DAILY_LOSS_USD=10000
MAX_POSITION_SIZE_BTC=1.0

# System Tuning
FPS_THROTTLE=60
WEBSOCKET_BACKPRESSURE_THRESHOLD=10000
```

Encrypt your `.env` file:
```bash
python deployment/security/env_encryptor.py encrypt .env
```

---

### 🎮 Operating the Dashboard

#### Main Layout Zones

```
┌────────────────────────────────────────────────────────────┐
│  HEADER: System Health | Connection Status | PnL Summary   │
├──────────────┬───────────────────────┬─────────────────────┤
│              │                       │                     │
│  ORDER BOOK  │    PRICE CHART        │   EXECUTION PANEL   │
│   HEATMAP    │  (WebGL Candlesticks) │  (Order Ticket)     │
│   (WebGL)    │    + Footprint        │                     │
│              │                       │                     │
├──────────────┼───────────────────────┼─────────────────────┤
│              │                       │                     │
│   POSITIONS  │    CVD / DELTA        │   STRATEGY GRID     │
│    GRID      │    (Canvas)           │   (Toggle/Risk)     │
│              │                       │                     │
└──────────────┴───────────────────────┴─────────────────────┘
```

#### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `ESC` | Cancel order entry |
| `CTRL+K` | Kill switch (confirm required) |
| `CTRL+P` | Pause strategy execution |
| `F1-F4` | Switch chart timeframes |
| `ALT+1-9` | Quick position sizing (10%-90%) |

#### Audio Feedback

The system provides subtle audio cues:
- **High click (2.8kHz → 1.2kHz):** Full order fill
- **Soft click (1.8kHz → 1kHz):** Partial fill
- **Low buzz (180Hz):** Order rejected
- **Two-tone chime:** System alert
- **Pulsing tone (600Hz):** Liquidation warning

Adjust volume in Settings → Audio.

---

### 🚨 Emergency Procedures

#### Kill Switch (Panic Button)

Located in the top-right corner of the dashboard. Requires **double-click + slide-to-confirm** to prevent accidental activation.

**What it does:**
1. Sends `EMERGENCY_FLATTEN` command to Rust backend
2. Cancels ALL open orders across all venues
3. Market-closes ALL positions at best available price
4. Pauses all strategy execution
5. Logs event to immutable audit trail

#### Manual Override

If the kill switch fails:
```bash
# Direct API call
curl -X POST https://localhost:8080/api/v1/emergency/flatten \
  -H "Authorization: Bearer <token>" \
  -d '{"reason": "manual_emergency"}'

# Or run teardown script
./deployment/teardown_master.sh
```

#### Circuit Breakers (Automatic)

The system auto-triggers when:
- Daily loss exceeds `MAX_DAILY_LOSS_USD`
- Drawdown > 15% from peak equity
- Latency spikes > 50ms for 5 consecutive seconds
- Exchange disconnect > 3 seconds

---

### ⚡ Performance Tuning

#### CPU Governor

For maximum performance:
```bash
sudo cpupower frequency-set -g performance
watch -n 1 'cpupower frequency-info'
```

For power saving:
```bash
sudo cpupower frequency-set -g powersave
```

#### Hugepages Allocation

Check current allocation:
```bash
cat /proc/sys/vm/nr_hugepages
```

Increase (requires root):
```bash
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
```

#### Docker Resource Monitoring

```bash
# Real-time memory/CPU usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check if hitting limits
docker inspect rust-core | grep -A5 "Memory"
```

#### Frontend FPS Throttling

The dashboard auto-throttles based on backend telemetry. Manual override:
```typescript
// In browser console
window.__SET_FPS_TARGET__(30); // Reduce to 30fps for battery saving
```

---

### 🐛 Troubleshooting

#### Issue: Frontend not loading

```bash
# Check SSL certificates
ls -la deployment/security/certs/

# Regenerate if expired
./deployment/security/mkcert_setup.sh

# Rebuild frontend
cd frontend && npm run build
```

#### Issue: High memory usage (>14GB)

```bash
# Identify culprit
docker stats

# If Python layer is high, reduce Ray workers
# Edit docker-compose.yml: ray_python -> environment -> RAY_WORKERS=2

# Restart affected service
docker compose restart python-ray
```

#### Issue: WebSocket disconnects

1. Check firewall rules:
   ```bash
   sudo ufw status verbose
   ```

2. Verify backend health:
   ```bash
   curl -k https://localhost:8080/health
   ```

3. Check exchange connectivity logs:
   ```bash
   docker compose logs rust-core | grep "exchange_gateway"
   ```

#### Issue: Chart rendering stutter

1. Reduce FPS target in Settings
2. Disable CRT scanline effect (`cyberpunk-polish.css`)
3. Clear browser cache (Ctrl+Shift+Delete)
4. Ensure hardware acceleration is enabled in browser

---

### 📞 Support & Maintenance

#### Log Collection

```bash
# Export last 24 hours of logs
docker compose logs --since 24h > bot_logs_$(date +%Y%m%d).txt
```

#### Database Backup

```bash
# Backup LMDB state
docker cp rust-core:/app/data/lmdb ./backups/lmdb_$(date +%Y%m%d)

# Backup trade journal (DuckDB)
docker cp python-ray:/app/data/journal.duckdb ./backups/
```

#### Version Updates

```bash
git pull origin main
docker compose down
./launch_master.sh
```

---

**Built with ❤️ for ultra-low-latency trading.**
*AMD Ryzen AI 5 + Radeon GPU Optimized | <14GB RAM Enforced | No LLMs*
