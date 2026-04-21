param(
  [string]$Competition = "byu-locating-bacterial-flagellar-motors-2025",
  [string]$OutDir = ".\data\kaggle\raw"
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$zipPath = Join-Path $OutDir "$Competition.zip"

$kaggleExe = Join-Path $PSScriptRoot "..\.venv\Scripts\kaggle.exe"
if (Test-Path $kaggleExe) {
  & $kaggleExe competitions download -c $Competition -p $OutDir
} else {
  kaggle competitions download -c $Competition -p $OutDir
}

if (Test-Path $zipPath) {
  Expand-Archive -Path $zipPath -DestinationPath $OutDir -Force
}
