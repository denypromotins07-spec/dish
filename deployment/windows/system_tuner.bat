@echo off
REM ============================================================================
REM Windows System Tuner for Crypto Trading Bot
REM ============================================================================
REM 
REM This master optimization script configures Windows for ultra-low latency:
REM - Disables Windows Core Parking
REM - Sets Power Plan to "Ultimate Performance"
REM - Disables Windows Search indexing on bot directories
REM - Adds bot folders to Windows Defender exclusions
REM - Optimizes network stack for low latency
REM
REM Target: AMD Ryzen AI 5, Windows 10/11, 16GB RAM (10GB bot limit)
REM ============================================================================

setlocal enabledelayedexpansion

REM Check for administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] This script must be run as Administrator!
    echo Right-click Command Prompt and select 'Run as Administrator'
    pause
    exit /b 1
)

echo ============================================================================
echo Windows System Tuner for Crypto Trading Bot
echo ============================================================================
echo.

REM ============================================================================
REM Configuration - Bot installation paths
REM ============================================================================
set "BOT_ROOT=C:\crypto_bot"
set "BOT_DATA=%BOT_ROOT%\data"
set "BOT_LOGS=%BOT_ROOT%\logs"

echo [INFO] Bot root directory: %BOT_ROOT%
echo.

REM ============================================================================
REM 1. Enable Ultimate Performance Power Plan
REM ============================================================================
echo [STEP 1/6] Configuring power plan...

REM Reveal the hidden Ultimate Performance power plan
powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61

REM Set it as active
powercfg -setactive e9a42b02-d5df-448d-aa00-03f14749eb61

REM Configure additional power settings
REM Set processor minimum state to 100% (prevents frequency scaling)
powercfg -setacvalueindex scheme_current sub_processor PROCMIN 100
powercfg -setacvalueindex scheme_current sub_processor PROCMAX 100

REM Disable processor idle states
powercfg -setacvalueindex scheme_current sub_processor IDLEDEMOTE 0

REM Apply changes
powercfg -setactive scheme_current

echo [OK] Ultimate Performance power plan activated
echo [OK] Processor frequency scaling disabled
echo.

REM ============================================================================
REM 2. Disable Windows Core Parking
REM ============================================================================
echo [STEP 2/6] Disabling Windows Core Parking...

REM Create registry entries to disable core parking
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583" /v ValueMax /t REG_DWORD /d 0 /f >nul 2>&1
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583" /v ValueMin /t REG_DWORD /d 0 /f >nul 2>&1

REM Unhide the setting
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583" /v Attributes /t REG_DWORD /d 2 /f >nul 2>&1

echo [OK] Core Parking disabled
echo.

REM ============================================================================
REM 3. Disable Windows Search Indexing on Bot Directories
REM ============================================================================
echo [STEP 3/6] Disabling Windows Search indexing...

REM Ensure bot directories exist
if not exist "%BOT_ROOT%" mkdir "%BOT_ROOT%"
if not exist "%BOT_DATA%" mkdir "%BOT_DATA%"
if not exist "%BOT_LOGS%" mkdir "%BOT_LOGS%"

REM Use icacls to disable indexing (mark as not content indexed)
echo D | icacls "%BOT_ROOT%" /grant administrators:F /t /c >nul 2>&1

REM Disable indexing attribute on directories
fsutil behavior set disableCompression 1 >nul 2>&1

echo [OK] Bot directories excluded from indexing
echo.

REM ============================================================================
REM 4. Add Windows Defender Exclusions
REM ============================================================================
echo [STEP 4/6] Adding Windows Defender exclusions...

REM Add folder exclusions (prevents real-time scanning delays)
powershell -Command "Add-MpPreference -ExclusionPath '%BOT_ROOT%' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionPath '%BOT_DATA%' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionPath '%BOT_LOGS%' -Force" 2>nul

REM Add process exclusions for bot executables
powershell -Command "Add-MpPreference -ExclusionProcessName 'python' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionProcessName 'rust_bot' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionProcessName 'msedge' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionProcessName 'chrome' -Force" 2>nul

REM Add extension exclusions for data files
powershell -Command "Add-MpPreference -ExclusionExtension '.parquet' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionExtension '.sqlite' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionExtension '.db' -Force" 2>nul
powershell -Command "Add-MpPreference -ExclusionExtension '.log' -Force" 2>nul

echo [OK] Windows Defender exclusions added
echo.

REM ============================================================================
REM 5. Optimize Network Stack for Low Latency
REM ============================================================================
echo [STEP 5/6] Optimizing network stack...

REM Disable TCP Nagle's algorithm (reduces latency)
reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces" /v TcpAckFlag /t REG_DWORD /d 1 /f >nul 2>&1
reg add "HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces" /v TCPNoDelay /t REG_DWORD /d 1 /f >nul 2>&1

REM Increase TCP window size for high throughput
netsh int tcp set global autotuninglevel=normal >nul 2>&1

REM Disable ECN (can cause latency spikes)
netsh int tcp set global ecncapability=disabled >nul 2>&1

REM Set network throttling to maximum
netsh int tcp set global netdma=enabled >nul 2>&1

echo [OK] Network stack optimized for low latency
echo.

REM ============================================================================
REM 6. Disable Unnecessary Windows Services
REM ============================================================================
echo [STEP 6/6] Disabling unnecessary services...

REM Stop and disable services that can cause latency spikes
sc stop "SysMain" >nul 2>&1
sc config "SysMain" start= disabled >nul 2>&1

sc stop "WSearch" >nul 2>&1
sc config "WSearch" start= disabled >nul 2>&1

sc stop "DiagTrack" >nul 2>&1
sc config "DiagTrack" start= disabled >nul 2>&1

sc stop "dmwappushservice" >nul 2>&1
sc config "dmwappushservice" start= disabled >nul 2>&1

echo [OK] Unnecessary services disabled
echo.

REM ============================================================================
REM Summary
REM ============================================================================
echo ============================================================================
echo System Optimization Complete!
echo ============================================================================
echo.
echo The following optimizations have been applied:
echo   [1] Ultimate Performance power plan activated
echo   [2] Core Parking disabled
echo   [3] Windows Search indexing disabled on bot directories
echo   [4] Windows Defender exclusions added
echo   [5] Network stack optimized for low latency
echo   [6] Unnecessary services disabled
echo.
echo IMPORTANT: A system restart is required for all changes to take effect!
echo.
echo To revert these changes, run: system_tuner_revert.bat
echo.
set /p restart="Restart now? (Y/N): "
if /i "%restart%"=="Y" (
    echo Restarting in 5 seconds...
    timeout /t 5 /nobreak >nul
    shutdown /r /t 0
) else (
    echo Please restart your computer manually for changes to take effect.
)

echo.
echo Press any key to exit...
pause >nul

exit /b 0
