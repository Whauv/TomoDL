param(
  [string]$RepoId = "Floppanacci/tomogram-Bacterial-Flagellar-motors-location",
  [string]$HFRoot = ".\data\hf_mirror\raw",
  [string]$OutputDir = ".\data",
  [double]$ValRatio = 0.15,
  [int]$Seed = 42,
  [switch]$Download
)

$downloadFlag = ""
if ($Download) {
  $downloadFlag = "--download"
}

$pythonExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (!(Test-Path $pythonExe)) {
  $pythonExe = "python"
}

& $pythonExe .\data\prepare_hf_mirror_data.py `
  --repo-id $RepoId `
  --hf-root $HFRoot `
  --output-dir $OutputDir `
  --val-ratio $ValRatio `
  --seed $Seed `
  $downloadFlag
