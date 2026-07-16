# Chapter 4, File 3: Windows HFT Tuner PowerShell Script
# scripts/windows_hft_tuner.ps1
# Master script for Windows HFT optimization

param(
    [switch]$Apply,
    [switch]$Revert,
    [switch]$Status,
    [string]$LogPath = "$env:TEMP\hft_tuner.log"
)

$ErrorActionPreference = "Stop"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Write-Log {
    param([string]$Message)
    $logEntry = "[$Timestamp] $Message"
    Add-Content -Path $LogPath -Value $logEntry
    Write-Host $Message
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Set-UltimatePerformancePowerPlan {
    Write-Log "[POWER] Setting Ultimate Performance power plan..."
    
    # Check if Ultimate Performance plan exists, create if not
    $guid = (powercfg /list | Select-String "Ultimate Performance").Line
    if ($null -eq $guid) {
        # Create Ultimate Performance scheme
        powercfg /duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61
        Write-Log "[POWER] Created Ultimate Performance power plan"
    }
    
    # Get the GUID of Ultimate Performance
    $ultimateGuid = (powercfg /list | Select-String "Ultimate Performance" | 
        ForEach-Object { ($_ -split '\s+')[3] })[0]
    
    if ($ultimateGuid) {
        powercfg /setactive $ultimateGuid
        Write-Log "[POWER] Activated Ultimate Performance plan: $ultimateGuid"
    } else {
        Write-Log "[WARNING] Could not find Ultimate Performance plan"
    }
    
    # Additional power optimizations
    powercfg /change /standby-timeout-ac 0
    powercfg /change /hibernate-timeout-ac 0
    powercfg /change /monitor-timeout-ac 0
    powercfg /change /disk-timeout-ac 0
    
    Write-Log "[POWER] Disabled all sleep timeouts"
}

function Disable-CoreParking {
    Write-Log "[CORE] Disabling core parking..."
    
    $registryPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583"
    
    if (Test-Path $registryPath) {
        Set-ItemProperty -Path $registryPath -Name "ValueMax" -Value 0 -Force
        Write-Log "[CORE] Core parking disabled via registry"
    } else {
        Write-Log "[WARNING] Registry path not found: $registryPath"
    }
}

function Disable-SysMain {
    Write-Log "[SYSMAIN] Disabling SysMain (SuperFetch) service..."
    
    Stop-Service -Name "SysMain" -Force -ErrorAction SilentlyContinue
    Set-Service -Name "SysMain" -StartupType Disabled -ErrorAction SilentlyContinue
    
    Write-Log "[SYSMAIN] SysMain service disabled"
}

function Disable-WindowsUpdateBackgroundTasks {
    Write-Log "[UPDATE] Disabling Windows Update background tasks during market hours..."
    
    # Create scheduled task to block updates during market hours (9:30 AM - 4:00 PM ET)
    $taskName = "HFT_BlockWindowsUpdate"
    
    # Check if task exists
    $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    
    if ($null -eq $existingTask) {
        # Create action to pause Windows Update
        $action = New-ScheduledTaskAction -Execute "powershell.exe" `
            -Argument "-Command `"Get-Service wuauserv | Stop-Service -Force`""
        
        # Create trigger for market open (9:25 AM)
        $triggerOpen = New-ScheduledTaskTrigger -Daily -At 9:25AM
        
        # Create trigger for market close (4:05 PM)
        $triggerClose = New-ScheduledTaskTrigger -Daily -At 4:05PM `
            -Execute "powershell.exe" `
            -Argument "-Command `"Get-Service wuauserv | Start-Service`""
        
        Register-ScheduledTask -TaskName $taskName `
            -Action $action `
            -Trigger $triggerOpen `
            -Description "Pause Windows Update during HFT market hours" `
            -RunLevel Highest `
            -Force
        
        Write-Log "[UPDATE] Created scheduled task to manage Windows Update"
    } else {
        Write-Log "[UPDATE] Windows Update management task already exists"
    }
}

function Optimize-NetworkForHFT {
    Write-Log "[NETWORK] Optimizing network stack for HFT..."
    
    # Disable TCP Auto-Tuning
    netsh int tcp set global autotuninglevel=disabled 2>$null
    Write-Log "[NETWORK] Disabled TCP auto-tuning"
    
    # Enable ECN capability
    netsh int tcp set global ecncapability=enabled 2>$null
    
    # Set initial RTO
    netsh int tcp set global initialRto=3000 2>$null
    
    # Disable Window Scaling (may reduce throughput but improves latency consistency)
    # Note: Uncomment only if needed for your specific exchange
    # netsh int tcp set global windowscaling=disabled
    
    Write-Log "[NETWORK] Network optimizations applied"
}

function Set-ProcessAffinityMask {
    Write-Log "[AFFINITY] Configuring processor affinity mask..."
    
    # Set IRQ affinity for network cards (advanced)
    # This requires knowing your NIC's IRQ number
    Write-Log "[AFFINITY] Manual IRQ affinity configuration recommended"
}

function Apply-AllOptimizations {
    Write-Log "=========================================="
    Write-Log "Applying ALL Windows HFT Optimizations"
    Write-Log "=========================================="
    
    if (-not (Test-Administrator)) {
        Write-Log "[ERROR] Must run as Administrator!"
        throw "Administrator privileges required"
    }
    
    try {
        Set-UltimatePerformancePowerPlan
        Disable-CoreParking
        Disable-SysMain
        Disable-WindowsUpdateBackgroundTasks
        Optimize-NetworkForHFT
        Set-ProcessAffinityMask
        
        Write-Log "=========================================="
        Write-Log "ALL OPTIMIZATIONS APPLIED SUCCESSFULLY"
        Write-Log "REBOOT RECOMMENDED FOR FULL EFFECT"
        Write-Log "=========================================="
    } catch {
        Write-Log "[ERROR] Optimization failed: $_"
        throw
    }
}

function Revert-AllOptimizations {
    Write-Log "=========================================="
    Write-Log "Reverting Windows HFT Optimizations"
    Write-Log "=========================================="
    
    if (-not (Test-Administrator)) {
        Write-Log "[ERROR] Must run as Administrator!"
        throw "Administrator privileges required"
    }
    
    try {
        # Re-enable core parking
        $registryPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583"
        if (Test-Path $registryPath) {
            Set-ItemProperty -Path $registryPath -Name "ValueMax" -Value 100 -Force
            Write-Log "[CORE] Core parking restored to default"
        }
        
        # Re-enable SysMain
        Set-Service -Name "SysMain" -StartupType Automatic -ErrorAction SilentlyContinue
        Start-Service -Name "SysMain" -ErrorAction SilentlyContinue
        Write-Log "[SYSMAIN] SysMain service re-enabled"
        
        # Remove Windows Update task
        Unregister-ScheduledTask -TaskName "HFT_BlockWindowsUpdate" -Confirm:$false -ErrorAction SilentlyContinue
        Write-Log "[UPDATE] Removed Windows Update management task"
        
        # Restore TCP auto-tuning
        netsh int tcp set global autotuninglevel=normal 2>$null
        Write-Log "[NETWORK] TCP auto-tuning restored to normal"
        
        # Restore balanced power plan
        powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e 2>$null
        Write-Log "[POWER] Restored Balanced power plan"
        
        Write-Log "=========================================="
        Write-Log "ALL OPTIMIZATIONS REVERTED"
        Write-Log "=========================================="
    } catch {
        Write-Log "[ERROR] Revert failed: $_"
        throw
    }
}

function Get-OptimizationStatus {
    Write-Log "=========================================="
    Write-Log "Current HFT Optimization Status"
    Write-Log "=========================================="
    
    # Power plan
    $currentPlan = powercfg /getactivescheme
    Write-Log "[POWER] Active power plan: $currentPlan"
    
    # Core parking status
    try {
        $parkingReg = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583" -Name "ValueMax" -ErrorAction SilentlyContinue
        if ($parkingReg.ValueMax -eq 0) {
            Write-Log "[CORE] Core parking: DISABLED (optimized)"
        } else {
            Write-Log "[CORE] Core parking: ENABLED (default)"
        }
    } catch {
        Write-Log "[CORE] Could not determine core parking status"
    }
    
    # SysMain status
    $sysmainService = Get-Service -Name "SysMain" -ErrorAction SilentlyContinue
    if ($sysmainService.Status -eq "Stopped") {
        Write-Log "[SYSMAIN] SysMain: DISABLED (optimized)"
    } else {
        Write-Log "[SYSMAIN] SysMain: RUNNING (default)"
    }
    
    # TCP auto-tuning
    $tcpStatus = netsh int tcp show global | Select-String "Auto-Tuning Level"
    Write-Log "[NETWORK] $tcpStatus"
    
    Write-Log "=========================================="
}

# Main execution
try {
    if ($Apply) {
        Apply-AllOptimizations
    } elseif ($Revert) {
        Revert-AllOptimizations
    } elseif ($Status) {
        Get-OptimizationStatus
    } else {
        Write-Host "Windows HFT Tuner - AMD Ryzen AI Optimization Script"
        Write-Host ""
        Write-Host "Usage:"
        Write-Host "  .\windows_hft_tuner.ps1 -Apply   # Apply all optimizations"
        Write-Host "  .\windows_hft_tuner.ps1 -Revert  # Revert to defaults"
        Write-Host "  .\windows_hft_tuner.ps1 -Status  # Show current status"
        Write-Host ""
        Write-Host "NOTE: Must run as Administrator!"
    }
} catch {
    Write-Log "[FATAL] Script failed: $_"
    exit 1
}
