# deploy_superpoint_model.ps1 - deploy SuperPoint ONNX model to app run dirs
#
# Background: SuperPoint descriptor plugin needs an ONNX model
# (superpoint_dynamic.onnx, dynamic H/W shape) at runtime, scanned from
# applicationDirPath()/models. The model (~5 MB) is NOT committed to git;
# this script copies it from the superpointc project into the target
# executable dir's models/.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deploy_superpoint_model.ps1 [-SourceFile <onnx>] [-TargetAppDir <dir>]
# Defaults:
#   SourceFile   = D:/project/python/superpointc/model/superpoint_dynamic.onnx
#   TargetAppDir = <repo>/build/src/app/release
param(
    [string]$SourceFile = "D:/project/python/superpointc/model/superpoint_dynamic.onnx",
    [string]$TargetAppDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $TargetAppDir) {
    $TargetAppDir = Join-Path (Split-Path $PSScriptRoot -Parent) "build/src/app/release"
}

if (-not (Test-Path $SourceFile)) {
    Write-Error "Source model not found: $SourceFile (use -SourceFile to override)"
}

$targetModels = Join-Path $TargetAppDir "models"
New-Item -ItemType Directory -Path $targetModels -Force | Out-Null

Copy-Item -Path $SourceFile -Destination $targetModels -Force
Write-Host ("deployed model: {0} ({1} KB) -> {2}" -f (Split-Path $SourceFile -Leaf), [math]::Round((Get-Item $SourceFile).Length/1KB), $targetModels)
