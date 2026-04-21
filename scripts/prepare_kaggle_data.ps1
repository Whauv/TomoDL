param(
  [string]$KaggleRoot = ".\data\kaggle\raw",
  [string]$OutputDir = ".\data",
  [double]$ValRatio = 0.15,
  [int]$Seed = 42
)

$pythonExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (!(Test-Path $pythonExe)) {
  $pythonExe = "python"
}

& $pythonExe .\data\prepare_kaggle_data.py `
  --kaggle-root $KaggleRoot `
  --output-dir $OutputDir `
  --val-ratio $ValRatio `
  --seed $Seed
