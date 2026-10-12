# recon_cli —— 稀疏重建命令行验证 / 基准 harness（阶段 0）
#
# 用途：不启动 GUI，直接对某个图像目录跑完整的稀疏链路
#       （特征提取 → 匹配 → 初始化 → 增量重建），打印注册相机数、
#       三维点数与每张图像的注册状态；可选写出 metrics JSON。
#
# 用法：recon_cli.exe <图像目录> [最大图像数] [匹配模式]
#                  [--metrics-out path.json] [--dataset-id name]

TEMPLATE = app
CONFIG  += console
CONFIG  -= app_bundle
TARGET   = recon_cli

# 启用算法链依赖（Ceres/PCL/Open3D），与 core 保持一致
SFM3_ALG_DEPS = 1

include(../../../common.pri)

INCLUDEPATH += ../../core
INCLUDEPATH += ../common

LIBS += -L$$OUT_PWD/../../core -lsfmcore

win32 {
    LIBS += -lpsapi
}

SOURCES += main.cpp
