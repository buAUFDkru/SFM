# deploy_xfeat_models.ps1 - deploy XFeat ONNX models to app/test run dirs
#
# Background: XFeat descriptor plugin needs ONNX model files (xfeat{H}x{W}.onnx)
# at runtime. Models are large (4-14 MB each, 8 files), so they are NOT committed
# to git. This script copies them from a source dir (e.g. sfm2.0 models) into the
# target executable dir's models/. XFeat scans applicationDirPath()/models.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deploy_xfeat_models.ps1 [-SourceDir <dir>] [-TargetAppDir <dir>]
# Defaults:
#   SourceDir    = D:/project/qt/sfm2.0/src/resources/models
#   TargetAppDir = <repo>/build/src/app/release
param(
    [string]$SourceDir = "D:/project/qt/sfm2.0/src/resources/models",
    [string]$TargetAppDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $TargetAppDir) {
    $TargetAppDir = Join-Path (Split-Path $PSScriptRoot -Parent) "build/src/app/release"
}

if (-not (Test-Path $SourceDir)) {
    Write-Error "Source model dir not found: $SourceDir (use -SourceDir to override)"
}

$targetModels = Join-Path $TargetAppDir "models"
New-Item -ItemType Directory -Path $targetModels -Force | Out-Null

$files = Get-ChildItem -Path $SourceDir -Filter "xfeat*.onnx" -File
if ($files.Count -eq 0) {
    Write-Error "No xfeat*.onnx models found in: $SourceDir"
}

foreach ($f in $files) {
    Copy-Item -Path $f.FullName -Destination $targetModels -Force
    Write-Host ("deployed model: {0} ({1} KB)" -f $f.Name, [math]::Round($f.Length/1KB))
}

Write-Host ("done: {0} models -> {1}" -f $files.Count, $targetModels)