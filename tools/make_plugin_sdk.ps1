# make_plugin_sdk.ps1 - assemble the redistributable Plugin SDK
#
# Background: third-party plugin developers only need the binary contract
# (interface headers + sfmcore.lib + common.pri), the official template
# plugins, and the spec document — NOT the sfm3.0 source tree. This script
# assembles those pieces into SFM3-PluginSDK/ so it can be shipped together
# with a closed-source release (see 插件开发规范.md §9.3 闭源分发场景).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File make_plugin_sdk.ps1 [-CoreLib <sfmcore.lib>] [-OutputDir <dir>]
# Defaults:
#   CoreLib   = <repo>/build/Desktop_Qt_6_10_1_MSVC2022_64bit-Release/src/core/sfmcore.lib
#   OutputDir = <repo>/SFM3-PluginSDK
param(
    [string]$CoreLib = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent

if (-not $CoreLib) {
    $CoreLib = Join-Path $repoRoot "build/Desktop_Qt_6_10_1_MSVC2022_64bit-Release/src/core/sfmcore.lib"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "SFM3-PluginSDK"
}

if (-not (Test-Path $CoreLib)) {
    Write-Error "sfmcore.lib not found: $CoreLib (build the project first, or pass -CoreLib)"
}

# ── assemble layout ──────────────────────────────────────────────────────────
$includeDir = Join-Path $OutputDir "include/sfm3/plugin"
$libDir     = Join-Path $OutputDir "lib"
$priDir     = Join-Path $OutputDir "pri"
$tplDir     = Join-Path $OutputDir "templates"
$docDir     = Join-Path $OutputDir "doc"
foreach ($d in @($includeDir, $libDir, $priDir, $tplDir, $docDir)) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# 1. interface headers (the binary contract, spec appendix A)
$headers = @(
    "src/core/plugin/ISfmPlugin.h",
    "src/core/plugin/IDescriptorPlugin.h",
    "src/core/plugin/IMatcherPlugin.h",
    "src/core/plugin/ParamSchema.h",
    "src/core/plugin/PluginTypes.h",
    "src/core/plugin/PluginInfo.h"
)
foreach ($h in $headers) {
    $src = Join-Path $repoRoot $h
    if (-not (Test-Path $src)) { Write-Error "missing header: $src" }
    Copy-Item $src $includeDir -Force
}

# 2. link contract (spec red line #2)
Copy-Item $CoreLib $libDir -Force

# 3. shared compile config (spec appendix B)
Copy-Item (Join-Path $repoRoot "common.pri") $priDir -Force

# 4. official template plugins (reference implementations)
# 先清空 templates/，避免重复运行时嵌套出 templates/foo/foo/
if (Test-Path $tplDir) { Remove-Item $tplDir -Recurse -Force }
New-Item -ItemType Directory -Path $tplDir -Force | Out-Null
foreach ($tpl in @("sample_descriptor", "sample_matcher")) {
    $src = Join-Path $repoRoot "templates/plugins/$tpl"
    if (-not (Test-Path $src)) { Write-Error "missing template: $src" }
    Copy-Item $src (Join-Path $tplDir $tpl) -Recurse -Force
}

# 5. spec + workflow documents
# 用目录枚举避免 .ps1 源文件编码导致中文文件名字面量损坏
$pluginDocsDir = Join-Path $repoRoot "plugin-docs"
if (-not (Test-Path $pluginDocsDir)) {
    Write-Error "missing plugin-docs/ (restore 插件开发规范.md / 插件开发流程.md there)"
}
$copiedDocs = 0
Get-ChildItem $pluginDocsDir -Filter "*.md" | Where-Object {
    $_.Name -ne "SDK-README.md"
} | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $docDir $_.Name) -Force
    $copiedDocs++
}
if ($copiedDocs -lt 2) {
    Write-Error "expected at least 2 docs under plugin-docs/ (got $copiedDocs)"
}

# ── README：复制已校对的 UTF-8 文案（避免 PowerShell here-string / Set-Content 乱码）──
$readmeSrc = Join-Path $pluginDocsDir "SDK-README.md"
if (-not (Test-Path $readmeSrc)) {
    Write-Error "missing plugin-docs/SDK-README.md"
}
Copy-Item $readmeSrc (Join-Path $OutputDir "README.md") -Force

Write-Host ("Plugin SDK assembled: {0}" -f $OutputDir)
Get-ChildItem $OutputDir -Recurse -File | ForEach-Object { Write-Host ("  {0}" -f $_.FullName.Substring($OutputDir.Length + 1)) }
