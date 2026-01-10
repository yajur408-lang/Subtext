# PowerShell script to enable Windows Long Path support
# Run this as Administrator

Write-Host "Enabling Windows Long Path Support..." -ForegroundColor Green
Write-Host "This requires Administrator privileges." -ForegroundColor Yellow
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Enable long paths via registry
$registryPath = "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
$propertyName = "LongPathsEnabled"
$propertyValue = 1

try {
    $currentValue = Get-ItemProperty -Path $registryPath -Name $propertyName -ErrorAction SilentlyContinue
    
    if ($currentValue -and $currentValue.LongPathsEnabled -eq 1) {
        Write-Host "Long Path support is already enabled!" -ForegroundColor Green
    } else {
        Set-ItemProperty -Path $registryPath -Name $propertyName -Value $propertyValue -Type DWord
        Write-Host "Long Path support has been enabled!" -ForegroundColor Green
        Write-Host "You may need to restart your computer for changes to take effect." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done! If you changed the setting, please restart your computer." -ForegroundColor Green

