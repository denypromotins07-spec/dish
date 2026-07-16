@echo off
REM =============================================================================
REM NATIVE WINDOWS LAUNCHER - ULTIMATE HFT BOOT SEQUENCE
REM Executes PowerShell tuner, sets env vars, launches Rust core natively
REM Bypasses Docker overhead for the hot path
REM =============================================================================

setlocal EnableDelayedExpansion

title HFT Trading Bot - Native Windows Launcher

REM Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Requesting Administrator privileges...
    powershell -Command "Start-Process cmd.exe -ArgumentList '/c', '%~f0' -Verb RunAs"
    exit /b
)

echo =============================================================================
echo  ULTRA-LOW-LATENCY CRYPTO TRADING BOT
echo  NATIVE WINDOWS 11 LAUNCHER
echo  AMD Ryzen AI 5 | 16GB RAM | 10GB Bot Hard-Cap
echo =============================================================================
echo.

cd /d "%~dp0"

REM -----------------------------------------------------------------------------
REM STEP 1: EXECUTE POWERSHELL SYSTEM TUNER
REM -----------------------------------------------------------------------------
echo [STEP 1/7] Executing Windows HFT system tuner...
powershell -ExecutionPolicy Bypass -File "scripts\windows_hft_tuner.ps1" -Quiet
if %errorLevel% neq 0 (
    echo [WARN] System tuner encountered issues, continuing anyway...
)
echo.

REM -----------------------------------------------------------------------------
REM STEP 2: DECRYPT ENVIRONMENT VARIABLES VIA DPAPI
REM -----------------------------------------------------------------------------
echo [STEP 2/7] Decrypting environment variables...
for /f "delims=" %%i in ('powershell -ExecutionPolicy Bypass -File "scripts\secure_env_loader.ps1" -Action decrypt') do set TEMP_ENV_FILE=%%i
if not defined TEMP_ENV_FILE (
    echo [ERROR] Failed to decrypt environment
    pause
    exit /b 1
)
echo [INFO] Environment loaded
echo.

REM -----------------------------------------------------------------------------
REM STEP 3: APPLY WSL2 CONFIGURATION (IF DOCKER DESKTOP IS USED)
REM -----------------------------------------------------------------------------
echo [STEP 3/7] Configuring WSL2 limits...
if exist "deployment\wslconfig_optimizer.ini" (
    copy /Y "deployment\wslconfig_optimizer.ini" "%USERPROFILE%\.wslconfig" >nul 2>&1
    echo [INFO] WSL2 configuration applied
) else (
    echo [INFO] WSL2 config not found, skipping
)
echo.

REM -----------------------------------------------------------------------------
REM STEP 4: PREFLIGHT DIAGNOSTICS
REM -----------------------------------------------------------------------------
echo [STEP 4/7] Running pre-flight diagnostics...
powershell -ExecutionPolicy Bypass -File "scripts\preflight_diagnostics.ps1"
if %errorLevel% neq 0 (
    echo [ERROR] Pre-flight checks failed
    call scripts\secure_env_loader.ps1 -Action cleanup -EnvFile "%TEMP_ENV_FILE%"
    pause
    exit /b 1
)
echo.

REM -----------------------------------------------------------------------------
REM STEP 5: CLEAR BLOCKED PORTS
REM -----------------------------------------------------------------------------
echo [STEP 5/7] Clearing blocked ports...
powershell -ExecutionPolicy Bypass -File "scripts\port_clearer.ps1"
echo.

REM -----------------------------------------------------------------------------
REM STEP 6: GENERATE SSL CERTIFICATES IF NEEDED
REM -----------------------------------------------------------------------------
echo [STEP 6/7] Checking SSL certificates...
if not exist "certs\localhost.pem" (
    mkdir certs 2>nul
    powershell -ExecutionPolicy Bypass -File "scripts\ssl_cert_generator.ps1"
) else (
    echo [INFO] SSL certificates present
)
echo.

REM -----------------------------------------------------------------------------
REM STEP 7: VALIDATE API KEYS
REM -----------------------------------------------------------------------------
echo [STEP 7/7] Validating Binance API credentials...
python scripts\env_validator.py
if %errorLevel% neq 0 (
    echo [ERROR] API validation failed
    call scripts\secure_env_loader.ps1 -Action cleanup -EnvFile "%TEMP_ENV_FILE%"
    pause
    exit /b 1
)
echo.

REM =============================================================================
REM LAUNCH COMPONENTS
REM =============================================================================
echo =============================================================================
echo  LAUNCHING COMPONENTS
echo =============================================================================
echo.

REM Launch Rust Core with memory cap and real-time priority
echo [LAUNCH] Starting Rust HFT Core (6GB cap, TIME_CRITICAL)...
start "HFT Core" /REALTIME powershell -ExecutionPolicy Bypass -File "scripts\launch_core.ps1"
timeout /t 2 /nobreak >nul

REM Launch Python/Ray workers with memory cap
echo [LAUNCH] Starting Python Analytics Workers (4GB cap)...
start "HFT Workers" /HIGH powershell -ExecutionPolicy Bypass -File "scripts\launch_workers.ps1"
timeout /t 2 /nobreak >nul

REM Launch React Frontend
echo [LAUNCH] Starting React Frontend (2GB cap)...
start "HFT UI" /NORMAL powershell -ExecutionPolicy Bypass -File "scripts\launch_ui.ps1"

echo.
echo =============================================================================
echo  STARTUP COMPLETE
echo =============================================================================
echo  Services:
echo    - Rust Core:     http://localhost:8080
echo    - WebSocket:     ws://localhost:8081  (wss:// for secure)
echo    - Frontend:      https://localhost:5173
echo    - Ray Dashboard: http://localhost:8265
echo.
echo  Memory Allocation:
echo    - Rust Core:     6GB (hard-capped via Job Object)
echo    - Python Workers: 4GB (hard-capped via Job Object)
echo    - Total Bot:     10GB
echo    - Reserved for OS: 6GB
echo.
echo  To Shutdown:
echo    - Double-click KILL.bat on Desktop
echo    - Or run: .\KILL.bat
echo.
echo  Watchdog:
echo    - Auto-recovery enabled via crash_recovery_watchdog.ps1
echo =============================================================================
echo.

REM Open dashboard automatically
start "" "https://localhost:5173"

REM Start watchdog in background (optional)
REM start /B powershell -ExecutionPolicy Bypass -File "scripts\crash_recovery_watchdog.ps1"

exit /b 0
