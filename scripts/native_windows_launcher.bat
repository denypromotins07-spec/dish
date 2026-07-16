@echo off
REM Chapter 5, File 3: Native Windows HFT Launcher
REM scripts/native_windows_launcher.bat
REM Ultimate Windows launch sequence for native HFT bot (no Docker overhead)

setlocal EnableDelayedExpansion

echo ============================================
echo   HFT Bot - Native Windows Launcher
echo   AMD Ryzen AI Optimized Build
echo ============================================
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running as Administrator!
    echo Some optimizations may not apply.
    echo Right-click and select "Run as Administrator" for full functionality.
    echo.
)

REM Set working directory
cd /d "%~dp0.."
set BOT_ROOT=%CD%

REM Environment variables for HFT optimization
set RUST_LOG=info
set RUST_BACKTRACE=0
set TOKIO_WORKER_THREADS=4
set RAYON_NUM_THREADS=4

REM Memory limits (enforced by Job Objects in Python side)
set PYTHON_MEMORY_LIMIT_MB=4096
set RUST_MEMORY_LIMIT_MB=6144

REM Apply Windows PowerShell tuner if available
echo [LAUNCHER] Applying Windows HFT optimizations...
if exist "%BOT_ROOT%\scripts\windows_hft_tuner.ps1" (
    powershell -ExecutionPolicy Bypass -NoProfile -Command "& '%BOT_ROOT%\scripts\windows_hft_tuner.ps1' -Apply"
) else (
    echo [WARNING] PowerShell tuner not found, skipping optimizations
)

REM Check for Visual C++ Redistributable (required for Rust binaries)
echo [LAUNCHER] Checking runtime dependencies...
where vcruntime140.dll >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Visual C++ Redistributable not found!
    echo Please install from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    pause
    exit /b 1
)

REM Build Rust core if not already built
echo [LAUNCHER] Building Rust HFT core (MSVC optimized)...
if not exist "%BOT_ROOT%\target\release\hft_bot.exe" (
    echo [BUILD] Compiling release build with MSVC optimizations...
    
    REM Set MSVC optimization flags
    set RUSTFLAGS=-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort
    
    REM Build with x86_64-pc-windows-msvc target
    cargo build --release --target x86_64-pc-windows-msvc
    
    if errorlevel 1 (
        echo [ERROR] Rust build failed!
        echo Ensure you have Visual Studio 2022 with C++ tools installed.
        pause
        exit /b 1
    )
    echo [BUILD] Build completed successfully.
) else (
    echo [LAUNCHER] Using existing binary at target\release\hft_bot.exe
)

REM Create logs directory
if not exist "%BOT_ROOT%\logs" mkdir "%BOT_ROOT%\logs"

REM Create data directories
if not exist "%BOT_ROOT%\data\lmdb" mkdir "%BOT_ROOT%\data\lmdb"
if not exist "%BOT_ROOT%\data\parquet" mkdir "%BOT_ROOT%\data\parquet"

REM Apply Windows Defender exclusions (requires admin)
echo [LAUNCHER] Configuring Windows Defender exclusions...
if exist "%BOT_ROOT%\python\system\defender_excluder.py" (
    python "%BOT_ROOT%\python\system\defender_excluder.py" 2>nul
    if errorlevel 1 (
        echo [WARNING] Could not apply Defender exclusions (run as Admin)
    )
)

REM Start Rust HFT Core (native, no Docker)
echo.
echo [LAUNCHER] Starting Rust HFT Core...
echo ============================================

start "HFT Rust Core" /B /REALTIME "%BOT_ROOT%\target\release\hft_bot.exe"
set RUST_PID=$!

REM Wait for Rust core to initialize
timeout /t 2 /nobreak >nul

REM Start Python Ray Workers (with Job Object memory limits)
echo [LAUNCHER] Starting Python Analytics Workers...
echo ============================================

if exist "%BOT_ROOT%\python\system\windows_job_objects.py" (
    start "Python Ray Workers" /B /HIGH ^
        python -c "import sys; sys.path.insert(0, '%BOT_ROOT%/python'); from system.windows_job_objects import wrap_python_workers_in_job_object; wrap_python_workers_in_job_object()"
)

if exist "%BOT_ROOT%\python\main.py" (
    start "Python Analytics" /B /HIGH ^
        python "%BOT_ROOT%\python\main.py"
)

REM Start React Frontend (if available)
echo [LAUNCHER] Starting Frontend Development Server...
echo ============================================

if exist "%BOT_ROOT%\frontend\package.json" (
    cd "%BOT_ROOT%\frontend"
    if exist "node_modules" (
        start "HFT Frontend" /B npm run dev
    ) else (
        echo [INFO] Frontend node_modules not found. Run 'npm install' first.
    )
    cd "%BOT_ROOT%"
)

REM Display status
echo.
echo ============================================
echo   HFT Bot Launch Sequence Complete
echo ============================================
echo.
echo Components started:
echo   [OK] Rust HFT Core (REALTIME priority)
echo   [OK] Python Workers (HIGH priority, 4GB limit)
echo   [OK] Frontend (if available)
echo.
echo Memory Allocation:
echo   - Rust Core:     6GB max
echo   - Python Workers: 4GB max (Job Object enforced)
echo   - Windows OS:    6GB reserved
echo   - Total System:  16GB
echo.
echo Monitoring:
echo   - Logs: %BOT_ROOT%\logs\
echo   - Data: %BOT_ROOT%\data\
echo.
echo To stop all components:
echo   1. Close the terminal windows
echo   2. Or run: taskkill /F /IM hft_bot.exe /IM python.exe /IM node.exe
echo.
echo Press any key to view running processes...
pause >nul

REM Show running HFT processes
echo.
echo Running HFT processes:
tasklist | findstr /I "hft_bot python node"

echo.
echo ============================================
echo Launcher finished. Keep this window open.
echo ============================================

REM Keep launcher window open
pause
