# 10GB RAM Allocation Manifest

## Crypto Trading Bot - Windows Memory Architecture

**Hardware Target:** AMD Ryzen AI 5, AMD Radeon GPU, 16GB Total RAM  
**OS:** Windows 10/11  
**Storage:** NVMe SSD  
**Strict System Limit:** 10GB Maximum RAM Usage

---

## Memory Allocation Breakdown

| Component | Allocation | Percentage | Description |
|-----------|------------|------------|-------------|
| **Rust Core Engine** | 2.0 GB | 20% | Low-latency market data ingestion, order matching, WebSocket handling via IOCP |
| **Python/Nautilus** | 3.0 GB | 30% | Strategy execution, signal generation, portfolio management |
| **Databases/SSD Cache** | 2.0 GB | 20% | DuckDB (200MB limit), SQLite journal, mmap file buffers |
| **Frontend (Browser)** | 1.5 GB | 15% | Chrome/Edge with V8 heap limited to 512MB, WebGL textures capped at 256MB |
| **OS/Buffer Reserve** | 1.5 GB | 15% | Windows kernel, network buffers, filesystem cache |
| **TOTAL** | **10.0 GB** | **100%** | Hard ceiling for entire bot system |

```
┌─────────────────────────────────────────────────────────────────┐
│                    16GB TOTAL SYSTEM RAM                        │
├─────────────────────────────────────────────────────────────────┤
│  ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ←────── 10GB BOT LIMIT ──────→←──── 6GB OS RESERVE ─────────→  │
│                                                                   │
│  Bot Allocation:                                                  │
│  ├── Rust Core:      2.0 GB ████████                             │
│  ├── Python:         3.0 GB ████████████                         │
│  ├── Databases:      2.0 GB ████████                             │
│  ├── Frontend:       1.5 GB ██████                               │
│  └── OS Buffer:      1.5 GB ██████                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Per-Process Memory Limits

### 1. Rust Core (`rust_bot.exe`)
- **Hard Limit:** 2GB via Windows Job Object
- **Working Set:** Locked to physical RAM (4-8GB range)
- **Thread Priority:** `THREAD_PRIORITY_TIME_CRITICAL`
- **CPU Affinity:** Pinned to logical cores 0-5 (physical cores)

### 2. Python Process (`python.exe` / NautilusTrader)
- **Hard Limit:** 2GB via Windows Job Object
- **Soft Limit:** 3GB (includes child processes)
- **GC Threshold:** Aggressive collection at 90% of limit
- **Buffer Flush:** Every 100 events or 1 second

### 3. Database Processes
- **DuckDB:** 200MB hard memory limit
- **SQLite Journal:** WAL mode, aggressive checkpointing
- **mmap Files:** Only top 10 order book levels in RAM

### 4. Frontend Browser (Chrome/Edge)
- **V8 Heap:** 512MB (`--js-flags="--max-old-space-size=512"`)
- **GPU Memory:** 256MB texture limit
- **Total Browser:** 1.5GB cap via Job Object
- **FPS:** Adaptive (60fps normal, 30fps high volatility)

---

## Monitoring Instructions

### Using Windows Task Manager

1. **Open Task Manager** (`Ctrl+Shift+Esc`)
2. Go to **Details** tab
3. Add columns:
   - Right-click header → Select columns
   - Check: `Memory (active private working set)`, `Memory (private commit)`
4. Monitor these processes:
   - `rust_bot.exe` - Should stay under 2,048 MB
   - `python.exe` - Should stay under 3,072 MB
   - `msedge.exe` / `chrome.exe` - Should stay under 1,536 MB
   - `duckdb.exe` - Should stay under 200 MB

### Using RAMMap (Microsoft Sysinternals)

1. Download from: https://learn.microsoft.com/en-us/sysinternals/downloads/rammap
2. Run as Administrator
3. Key views to monitor:
   - **Use Counts**: Shows total memory breakdown by category
   - **Processes**: Per-process memory details
   - **File Summary**: mmap file usage on NVMe SSD
4. Watch for:
   - `Private Bytes` exceeding limits
   - `Mapped File` growth (should be on SSD, not RAM)

### Using PowerShell

```powershell
# Get memory usage for all bot-related processes
Get-Process | Where-Object {
    $_.Name -match 'python|rust_bot|msedge|chrome|duckdb'
} | Select-Object Name, @{N='Mem(MB)';E={[math]::Round($_.WorkingSet/1MB,2)}} | Sort-Object 'Mem(MB)' -Descending

# Check total bot memory
$botProcs = Get-Process | Where-Object { $_.Name -match 'python|rust_bot|msedge|chrome|duckdb' }
$totalMB = ($botProcs | Measure-Object WorkingSet -Sum).Sum / 1MB
Write-Host "Total Bot Memory: $([math]::Round($totalMB, 2)) MB / 10240 MB (10GB)"
```

---

## Emergency Memory Actions

When system RAM approaches 9.5GB (95% of limit):

### Automatic Triggers
1. **Python GC** forced immediately
2. **Nautilus buffers** flushed to SQLite
3. **Ray workers** paused
4. **Frontend FPS** reduced to 30

### Manual Interventions
1. Close unnecessary browser tabs
2. Run buffer flush: `python -m python.memory.ram_enforcer_win`
3. Restart frontend: Close and relaunch `launch_browser.bat`
4. Check for memory leaks in logs

---

## Configuration Files Reference

| File | Purpose | Memory Impact |
|------|---------|---------------|
| `crates/windows/src/win32_tuner.rs` | Win32 API memory locking | Locks 4-8GB working set |
| `crates/windows/src/iocp_network.rs` | IOCP network engine | Zero-copy, ~100MB |
| `crates/windows/src/mmap_ssd_engine.rs` | NVMe mmap storage | Top 10 levels only (~50MB) |
| `python/memory/ram_enforcer_win.py` | RAM watchdog | Triggers at 9.5GB |
| `python/data/out_of_core_duckdb.py` | Out-of-core processing | 200MB DuckDB limit |
| `python/nautilus/buffer_shrinker.py` | Buffer reduction | Saves ~1.5GB |
| `frontend/vite.config.ts` | Code splitting | Reduces initial load |
| `frontend/src/core/webgl_memory_manager.ts` | WebGL limits | 256MB texture cap |
| `frontend/public/launch_browser.bat` | Browser flags | 512MB V8 heap |
| `deployment/windows/process_isolator.py` | Job Objects | Hard per-process caps |
| `deployment/windows/system_tuner.bat` | System optimization | Prevents paging |

---

## Troubleshooting

### Problem: Process exceeds memory limit
**Solution:** Windows Job Object will automatically terminate the process. Check logs for OOM events.

### Problem: System becomes unresponsive
**Solution:** Reduce Rust core working set from 8GB to 6GB in `win32_tuner.rs`.

### Problem: High latency spikes
**Solution:** Verify Windows Defender exclusions are active. Run `system_tuner.bat` again.

### Problem: Browser crashes frequently
**Solution:** Lower V8 heap limit to 384MB in `launch_browser.bat`.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial 10GB architecture implementation |

---

**Document Owner:** Systems Engineering  
**Last Updated:** 2024  
**Classification:** Internal Use Only
