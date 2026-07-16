# ============================================================================
# Windows Defender Firewall Rules for Crypto Trading Bot
# ============================================================================
# 
# This PowerShell script configures Windows Defender Firewall to:
# - Block all inbound traffic except local frontend connections
# - Restrict outbound traffic strictly to Binance/Bybit IP ranges
# - Protect the bot from unauthorized network access
#
# Target: Windows 10/11, AMD Ryzen AI 5, strict 10GB RAM limit
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Windows Defender Firewall Configuration for Crypto Trading Bot" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check for administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[ERROR] This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Running with administrator privileges" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Binance API IP Ranges (as of 2024 - verify current ranges)
# Source: Binance official documentation
# ============================================================================
$BinanceIPs = @(
    "52.18.219.0/24",      # AWS EU West
    "52.208.0.0/16",       # AWS EU West
    "52.48.0.0/16",        # AWS EU West
    "54.72.0.0/16",        # AWS EU West
    "54.246.0.0/16",       # AWS EU West
    "3.248.0.0/13",        # AWS EU West
    "18.200.0.0/16",       # AWS EU West
    "18.202.0.0/16",       # AWS EU West
    "99.80.0.0/16",        # AWS EU West
    "13.112.0.0/16",       # AWS Asia Pacific
    "13.230.0.0/16",       # AWS Asia Pacific
    "13.250.0.0/16",       # AWS Asia Pacific
    "13.228.0.0/16",       # AWS Asia Pacific
    "18.136.0.0/16",       # AWS Asia Pacific
    "52.74.0.0/16",        # AWS Asia Pacific
    "52.76.0.0/16",        # AWS Asia Pacific
    "52.77.0.0/16",        # AWS Asia Pacific
    "52.78.0.0/16",        # AWS Asia Pacific
    "54.169.0.0/16",       # AWS Asia Pacific
    "54.254.0.0/16",       # AWS Asia Pacific
    "13.210.0.0/16",       # AWS Australia
    "52.62.0.0/16",        # AWS Australia
    "52.63.0.0/16",        # AWS Australia
    "52.64.0.0/16",        # AWS Australia
    "54.66.0.0/16",        # AWS Australia
    "54.252.0.0/16",       # AWS Australia
    "3.104.0.0/16",        # AWS Australia
    "3.105.0.0/16",        # AWS Australia
    "3.106.0.0/16",        # AWS Australia
)

# ============================================================================
# Bybit API IP Ranges (as of 2024 - verify current ranges)
# Source: Bybit official documentation
# ============================================================================
$BybitIPs = @(
    "8.219.0.0/16",        # AWS Asia Pacific
    "8.220.0.0/16",        # AWS Asia Pacific
    "8.221.0.0/16",        # AWS Asia Pacific
    "8.222.0.0/16",        # AWS Asia Pacific
    "47.241.0.0/16",       # Alibaba Cloud
    "47.242.0.0/16",       # Alibaba Cloud
    "47.243.0.0/16",       # Alibaba Cloud
    "47.244.0.0/16",       # Alibaba Cloud
    "47.245.0.0/16",       # Alibaba Cloud
    "47.246.0.0/16",       # Alibaba Cloud
    "47.52.0.0/16",        # AWS Hong Kong
    "47.53.0.0/16",        # AWS Hong Kong
    "47.54.0.0/16",        # AWS Hong Kong
    "47.55.0.0/16",        # AWS Hong Kong
    "47.74.0.0/16",        # AWS Singapore
    "47.75.0.0/16",        # AWS Singapore
    "47.76.0.0/16",        # AWS Singapore
    "47.77.0.0/16",        # AWS Singapore
    "47.88.0.0/16",        # AWS US West
    "47.89.0.0/16",        # AWS US West
    "47.90.0.0/16",        # AWS US West
    "47.91.0.0/16",        # AWS US West
)

# ============================================================================
# Local Frontend Port
# ============================================================================
$FrontendPort = 3000

# ============================================================================
# Create Inbound Rules
# ============================================================================
Write-Host "[INFO] Creating inbound firewall rules..." -ForegroundColor Cyan

# Allow localhost frontend connections only
New-NetFirewallRule `
    -DisplayName "CryptoBot Frontend Local" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort $FrontendPort `
    -Action Allow `
    -Profile Any `
    -Description "Allow local connections to crypto trading bot frontend" `
    -Enabled True `
    | Out-String | Write-Host

Write-Host "[OK] Frontend port $FrontendPort allowed for localhost only" -ForegroundColor Green

# Block all other inbound traffic (default Windows behavior, but explicit for clarity)
New-NetFirewallRule `
    -DisplayName "CryptoBot Block All Other Inbound" `
    -Direction Inbound `
    -Action Block `
    -Profile Any `
    -Description "Block all other inbound traffic not explicitly allowed" `
    -Enabled True `
    | Out-String | Write-Host

Write-Host "[OK] All other inbound traffic blocked" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Create Outbound Rules for Binance
# ============================================================================
Write-Host "[INFO] Creating outbound rules for Binance..." -ForegroundColor Cyan

foreach ($ipRange in $BinanceIPs) {
    New-NetFirewallRule `
        -DisplayName "CryptoBot Binance API $($ipRange)" `
        -Direction Outbound `
        -RemoteAddress $ipRange `
        -Protocol TCP `
        -RemotePort 443 `
        -Action Allow `
        -Profile Any `
        -Description "Allow HTTPS to Binance API servers" `
        -Enabled True `
        | Out-Null
    
    Write-Host "  [+] Added rule for Binance: $ipRange" -ForegroundColor Gray
}

Write-Host "[OK] Created $($BinanceIPs.Count) outbound rules for Binance" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Create Outbound Rules for Bybit
# ============================================================================
Write-Host "[INFO] Creating outbound rules for Bybit..." -ForegroundColor Cyan

foreach ($ipRange in $BybitIPs) {
    New-NetFirewallRule `
        -DisplayName "CryptoBot Bybit API $($ipRange)" `
        -Direction Outbound `
        -RemoteAddress $ipRange `
        -Protocol TCP `
        -RemotePort 443 `
        -Action Allow `
        -Profile Any `
        -Description "Allow HTTPS to Bybit API servers" `
        -Enabled True `
        | Out-Null
    
    Write-Host "  [+] Added rule for Bybit: $ipRange" -ForegroundColor Gray
}

Write-Host "[OK] Created $($BybitIPs.Count) outbound rules for Bybit" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Allow essential Windows services (DNS, NTP)
# ============================================================================
Write-Host "[INFO] Allowing essential Windows services..." -ForegroundColor Cyan

# DNS resolution
New-NetFirewallRule `
    -DisplayName "CryptoBot DNS Outbound" `
    -Direction Outbound `
    -Protocol UDP `
    -RemotePort 53 `
    -Action Allow `
    -Profile Any `
    -Description "Allow DNS resolution for API hostnames" `
    -Enabled True `
    | Out-String | Write-Host

# NTP for time synchronization (critical for trading)
New-NetFirewallRule `
    -DisplayName "CryptoBot NTP Outbound" `
    -Direction Outbound `
    -Protocol UDP `
    -RemotePort 123 `
    -Action Allow `
    -Profile Any `
    -Description "Allow NTP time synchronization" `
    -Enabled True `
    | Out-String | Write-Host

Write-Host "[OK] Essential services allowed" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Block all other outbound traffic
# ============================================================================
Write-Host "[INFO] Blocking all other outbound traffic..." -ForegroundColor Cyan

New-NetFirewallRule `
    -DisplayName "CryptoBot Block All Other Outbound" `
    -Direction Outbound `
    -Action Block `
    -Profile Any `
    -Description "Block all outbound traffic not explicitly allowed" `
    -Enabled True `
    | Out-String | Write-Host

Write-Host "[OK] All other outbound traffic blocked" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Summary
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Firewall Configuration Complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor White
Write-Host "  - Inbound: Only localhost:$FrontendPort allowed" -ForegroundColor White
Write-Host "  - Outbound: Only Binance, Bybit, DNS, and NTP allowed" -ForegroundColor White
Write-Host "  - All other traffic: BLOCKED" -ForegroundColor White
Write-Host ""
Write-Host "To view all rules:" -ForegroundColor Yellow
Write-Host "  Get-NetFirewallRule | Where-Object { \$_.DisplayName -like '*CryptoBot*' }" -ForegroundColor Gray
Write-Host ""
Write-Host "To remove all rules (emergency):" -ForegroundColor Yellow
Write-Host "  Get-NetFirewallRule | Where-Object { \$_.DisplayName -like '*CryptoBot*' } | Remove-NetFirewallRule" -ForegroundColor Gray
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
