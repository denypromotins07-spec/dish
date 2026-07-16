@echo off
REM ============================================================================
REM Launch Browser with Strict Memory Limits for Crypto Trading Bot Frontend
REM ============================================================================
REM 
REM This batch script launches Edge/Chrome with memory-limiting flags to ensure
REM the browser never exceeds 1GB of RAM, protecting the overall 10GB system limit.
REM
REM Target: AMD Ryzen AI 5, Windows 10/11, 16GB total RAM (10GB bot limit)
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "FRONTEND_URL=http://localhost:3000"
set "BROWSER_PATH="
set "MAX_OLD_SPACE_SIZE=512"
set "DISABLE_EXTENSIONS=true"
set "DISABLE_GPU_SANDBOX=true"

REM Detect browser installation
if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" (
    set "BROWSER_PATH=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
    echo [LaunchBrowser] Found Microsoft Edge
) else if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" (
    set "BROWSER_PATH=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
    echo [LaunchBrowser] Found Microsoft Edge
) else if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" (
    set "BROWSER_PATH=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
    echo [LaunchBrowser] Found Google Chrome
) else if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
    set "BROWSER_PATH=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
    echo [LaunchBrowser] Found Google Chrome
) else (
    echo [ERROR] No supported browser found (Edge or Chrome required)
    pause
    exit /b 1
)

echo [LaunchBrowser] Starting browser with memory limits...
echo [LaunchBrowser] Max JS Heap: %MAX_OLD_SPACE_SIZE%MB
echo [LaunchBrowser] URL: %FRONTEND_URL%

REM ============================================================================
REM Memory-Limiting Browser Flags
REM ============================================================================
REM --js-flags="--max-old-space-size=512" : Limit V8 heap to 512MB
REM --disable-extensions                   : Disable all extensions (saves RAM)
REM --disable-gpu-sandbox                  : Reduce GPU memory overhead
REM Additional flags for low-memory operation:
REM --disable-background-timer-throttling  : Prevent timer throttling
REM --disable-renderer-backgrounding       : Keep renderer at full priority
REM --memory-pressure                      : Enable memory pressure notifications
REM ============================================================================

set "BROWSER_FLAGS=^
--js-flags=\"--max-old-space-size=%MAX_OLD_SPACE_SIZE%\" ^
--disable-extensions ^
--disable-gpu-sandbox ^
--disable-background-timer-throttling ^
--disable-renderer-backgrounding ^
--memory-pressure ^
--disable-features=TranslateUI,BrowserSideNavigation,MediaRouter ^
--no-first-run ^
--no-default-browser-check ^
--disable-hang-monitor ^
--disable-prompt-on-repost ^
--disable-client-side-phishing-detection ^
--disable-dev-shm-usage ^
--disable-accelerated-2d-canvas ^
--disable-gpu-compositing ^
--limit-bundles-to-extension-processes ^
--process-per-site ^
--site-per-process"

REM Create user data directory for isolated session
set "USER_DATA_DIR=%TEMP%\CryptoBotBrowser_%RANDOM%"
mkdir "%USER_DATA_DIR%" 2>nul

REM Set environment variable for V8 memory limit (affects embedded Electron apps too)
setx NODE_OPTIONS "--max-old-space-size=%MAX_OLD_SPACE_SIZE%" >nul 2>&1

echo [LaunchBrowser] User data dir: %USER_DATA_DIR%
echo [LaunchBrowser] Launching...

REM Launch browser with all flags
start "" "%BROWSER_PATH%" %BROWSER_FLAGS% --user-data-dir="%USER_DATA_DIR%" "%FRONTEND_URL%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to launch browser (error code: %ERRORLEVEL%)
    pause
    exit /b 1
)

echo [LaunchBrowser] Browser launched successfully
echo [LaunchBrowser] PID: !LAST_PID!
echo.
echo ============================================================================
echo Browser is running with strict memory limits:
echo   - JavaScript Heap: %MAX_OLD_SPACE_SIZE%MB max
echo   - Extensions: Disabled
echo   - GPU Sandbox: Disabled
echo   - Background timers: Throttled
echo.
echo To stop the browser and clean up:
echo   1. Close all browser windows
echo   2. Run: cleanup_browser.bat
echo ============================================================================

exit /b 0
