# PowerShell script to help install Tesseract OCR on Windows

Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "Tesseract OCR Installation Helper for Windows" -ForegroundColor Yellow
Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host "Step 1: Download Tesseract OCR" -ForegroundColor Green
Write-Host "-" * 70
Write-Host ""
Write-Host "Please download Tesseract OCR from:"
Write-Host "https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Cyan
Write-Host ""
Write-Host "Recommended download:"
Write-Host "  tesseract-ocr-w64-setup-5.x.x.exe (64-bit)" -ForegroundColor Yellow
Write-Host ""
Write-Host "After downloading, run the installer and note the installation path."
Write-Host "Default path: C:\Program Files\Tesseract-OCR" -ForegroundColor Yellow
Write-Host ""

# Check if Tesseract is already installed
$tesseractPaths = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "$env:ProgramFiles\Tesseract-OCR\tesseract.exe",
    "$env:ProgramFiles(x86)\Tesseract-OCR\tesseract.exe"
)

$foundPath = $null
foreach ($path in $tesseractPaths) {
    if (Test-Path $path) {
        $foundPath = $path
        break
    }
}

if ($foundPath) {
    Write-Host "✓ Tesseract found at: $foundPath" -ForegroundColor Green
    Write-Host ""
    Write-Host "Step 2: Configure Python" -ForegroundColor Green
    Write-Host "-" * 70
    Write-Host ""
    Write-Host "Add this to your Python code or viewer.py:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "import pytesseract" -ForegroundColor Cyan
    Write-Host "pytesseract.pytesseract.tesseract_cmd = r'$foundPath'" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "✗ Tesseract not found in common locations" -ForegroundColor Red
    Write-Host ""
    Write-Host "Step 2: After Installation" -ForegroundColor Green
    Write-Host "-" * 70
    Write-Host ""
    Write-Host "1. Install Tesseract using the downloaded installer"
    Write-Host "2. Note the installation path"
    Write-Host "3. Add this to viewer.py:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "import pytesseract" -ForegroundColor Cyan
    Write-Host "pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Step 3: Add to PATH (Optional but Recommended)" -ForegroundColor Green
Write-Host "-" * 70
Write-Host ""
if (-not $isAdmin) {
    Write-Host "To add Tesseract to PATH, run PowerShell as Administrator and run:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host '$env:Path += ";C:\Program Files\Tesseract-OCR"' -ForegroundColor Cyan
    Write-Host "[Environment]::SetEnvironmentVariable('Path', `$env:Path, 'Machine')" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Would you like to add Tesseract to PATH? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq 'Y' -or $response -eq 'y') {
        if ($foundPath) {
            $tesseractDir = Split-Path $foundPath -Parent
            $currentPath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
            if ($currentPath -notlike "*$tesseractDir*") {
                [Environment]::SetEnvironmentVariable('Path', "$currentPath;$tesseractDir", 'Machine')
                Write-Host "✓ Added Tesseract to PATH" -ForegroundColor Green
            } else {
                Write-Host "✓ Tesseract already in PATH" -ForegroundColor Green
            }
        } else {
            Write-Host "Please install Tesseract first, then run this script again." -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "Done! Restart your Streamlit app after installation." -ForegroundColor Green
Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan

