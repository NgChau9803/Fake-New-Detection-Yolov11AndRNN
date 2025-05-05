# PowerShell Script to set up and run the Fake News Detection System

# Check for Python 3.11
$python = Get-Command python -ErrorAction SilentlyContinue
$py311 = Get-Command py -ErrorAction SilentlyContinue
$hasPy311 = $false
if ($py311) {
    $ver = py -3.11 --version 2>$null
    if ($ver -match 'Python 3.11') { $hasPy311 = $true }
}
if (-not $hasPy311 -and $python) {
    $ver = python --version 2>$null
    if ($ver -match '3.11') { $hasPy311 = $true }
}
if (-not $hasPy311) {
    Write-Host "Python 3.11 is required but not found. Please install Python 3.11 and try again." -ForegroundColor Red
    exit 1
}

# Set up virtual environment (if not already created)
if (-Not (Test-Path ".\.venv")) {
    Write-Host "Creating Python 3.11 virtual environment..." -ForegroundColor Green
    py -3.11 -m venv .venv
    if (-Not (Test-Path ".\.venv")) {
        Write-Host "Failed to create virtual environment. Ensure Python 3.11 and venv are installed." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
try {
    & .\.venv\Scripts\Activate.ps1
} catch {
    Write-Host "Failed to activate virtual environment. Try '.\.venv\Scripts\Activate.ps1' manually." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Green
python -m pip install -r requirements.txt

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Green
$directories = @(
    "data/raw", 
    "data/processed", 
    "data/images", 
    "data/cache", 
    "logs/tensorboard", 
    "models", 
    "explanations", 
    "config"
)

foreach ($dir in $directories) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Cyan
    }
}

# Check if src directory exists
if (-Not (Test-Path "src")) {
    Write-Host "Warning: 'src' directory not found. The pipeline may not run correctly." -ForegroundColor Yellow
}

# Display menu for pipeline options
function Show-Menu {
    Write-Host "`n========= Fake News Detection Pipeline =========" -ForegroundColor Magenta
    Write-Host "1. Run Data Preprocessing"
    Write-Host "2. Train Model"
    Write-Host "3. Evaluate Model"
    Write-Host "4. Generate Visualizations"
    Write-Host "5. Run Complete Pipeline"
    Write-Host "6. Exit"
    Write-Host "===============================================" -ForegroundColor Magenta
    Write-Host "Enter your choice (1-6): " -NoNewline -ForegroundColor Green
}

# Run the selected option
function Run-Option {
    param (
        [Parameter(Mandatory=$true)]
        [int]$Option
    )

    switch ($Option) {
        1 {
            Write-Host "`nRunning Data Preprocessing..." -ForegroundColor Cyan
            python -c "from src.data.dataset import DatasetProcessor; import yaml; with open('config/config.yaml', 'r') as f: config = yaml.safe_load(f); processor = DatasetProcessor(config); processor.process_datasets(); processor.combine_datasets(); processor.preprocess_dataset()"
        }
        2 {
            Write-Host "`nRunning Model Training..." -ForegroundColor Cyan
            python -m src.main --train
        }
        3 {
            Write-Host "`nRunning Model Evaluation..." -ForegroundColor Cyan
            python -m src.main --evaluate
        }
        4 {
            Write-Host "`nGenerating Visualizations..." -ForegroundColor Cyan
            python -m src.main --explain
        }
        5 {
            Write-Host "`nRunning Complete Pipeline..." -ForegroundColor Cyan
            python -m src.main --all
        }
        6 {
            Write-Host "`nExiting..." -ForegroundColor Yellow
            exit
        }
        default {
            Write-Host "`nInvalid option. Please try again." -ForegroundColor Red
        }
    }
}

# Main execution
Write-Host "To rerun this menu, use ./setup_and_run.ps1" -ForegroundColor Cyan

do {
    Show-Menu
    $choice = Read-Host
    if ($choice -match '^[1-6]$') {
        Run-Option -Option ([int]$choice)
        if ([int]$choice -eq 6) { break }
    } else {
        Write-Host "Invalid input. Please enter a number between 1 and 6." -ForegroundColor Red
    }
    Write-Host "`nPress Enter to continue..." -ForegroundColor Green
    Read-Host
} while ($true) 