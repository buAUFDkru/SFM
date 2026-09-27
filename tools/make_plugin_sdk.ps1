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
foreach ($tpl in @("sample_descriptor", "sample_matcher")) {
    $src = Join-Path $repoRoot "src/plugins/$tpl"
    if (-not (Test-Path $src)) { Write-Error "missing template: $src" }
    Copy-Item $src (Join-Path $tplDir $tpl) -Recurse -Force
}

# 5. spec + workflow documents
Copy-Item (Join-Path $repoRoot "插件开发规范.md") $docDir -Force
if (Test-Path (Join-Path $repoRoot "插件开发流程.md")) {
    Copy-Item (Join-Path $repoRoot "插件开发流程.md") $docDir -Force
}

# ── README ───────────────────────────────────────────────────────────────────
$readme = @"
# SFM3 Plugin SDK

为 SFM Reconstructor ($([char]0x2265) v3.0) 开发算法插件所需的全部宿主产物。
配合 doc/插件开发规范.md 使用；规范第 9 章列出必须遵守的 ABI 红线。

## 目录结构

    include/sfm3/plugin/   宿主接口头文件（ISfmPlugin / IDescriptorPlugin / IMatcherPlugin /
                           ParamSchema / PluginTypes / PluginInfo）——二进制契约的声明部分
    lib/sfmcore.lib        接口 moc meta-object 链接库（规范红线 #2：禁止自行重编译接口头）
    pri/common.pri         与宿主一致的编译配置与第三方路径（OpenCV 460 / onnxruntime 1.26.0）
    templates/             官方模板插件：sample_descriptor（描述符）、sample_matcher（匹配器）
    doc/插件开发规范.md     插件开发规范（接口三件套、线程模型、ABI 红线、测试要求）
    doc/插件开发流程.md     插件开发流程（从零到可加载 DLL 的分步操作手册）

## 快速开始

1. 复制 templates/sample_descriptor 到你的工作目录，按规范第 3/4 节实现三件套
   （工厂 / Worker / plugin.json）。
2. .pro 中把宿主内路径改为 SDK 相对路径（两处差异）：

        # 宿主工程内写法                     →   SDK 离线开发写法
        INCLUDEPATH += ../../core            →   INCLUDEPATH += <SDK>/include/sfm3/plugin
        LIBS += -L$$OUT_PWD/../../core -lsfmcore →  LIBS += -L<SDK>/lib -lsfmcore

   其中 <SDK> 建议用相对路径变量引入，例如：
        SDK_ROOT = $$PWD/../SFM3-PluginSDK
        INCLUDEPATH += $$SDK_ROOT/include/sfm3/plugin
        LIBS += -L$$SDK_ROOT/lib -lsfmcore
   common.pri 同样从 $$SDK_ROOT/pri 引入。
3. 部署：构建出的 DLL 与其运行时依赖放到的目标机 plugins/ 目录须与宿主版本匹配
   （同 Qt 6.10.1 msvc2022_64、MSVC 2022 /MD release、OpenCV 460 world，规范红线 #4~#7）。
4. 验证：把 DLL 放入宿主 plugins/ 目录启动宿主，扫描日志出现
   "已加载插件: <name> (<category>)" 即加载成功；"非 ISfmPlugin" = IID/Q_INTERFACES 问题，
   "加载失败" = 运行时依赖问题（规范 §9.4）。

## 版本匹配

本 SDK 对应的宿主构建以 lib/sfmcore.lib 为准；宿主升级后必须使用配套版本重新发布
的 SDK 重新编译插件（IID 与跨 DLL 结构体布局是版本化契约，规范 §9.2）。
"@
Set-Content -Path (Join-Path $OutputDir "README.md") -Value $readme -Encoding UTF8

Write-Host ("Plugin SDK assembled: {0}" -f $OutputDir)
Get-ChildItem $OutputDir -Recurse -File | ForEach-Object { Write-Host ("  {0}" -f $_.FullName.Substring($OutputDir.Length + 1)) }
