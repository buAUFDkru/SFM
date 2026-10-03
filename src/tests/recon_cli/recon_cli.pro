# recon_cli —— 稀疏重建命令行验证工具
#
# 用途：不启动 GUI，直接对某个图像目录跑完整的稀疏链路
#       （特征提取 → 匹配 → 初始化 → 增量重建），打印注册相机数、
#       三维点数与每张图像的注册状态，用于在真实数据集上快速定位失败原因。
#
# 用法：recon_cli.exe <图像目录> [最大图像数] [匹配模式 auto|sequential|exhaustive]

TEMPLATE = app
CONFIG  += console
CONFIG  -= app_bundle
TARGET   = recon_cli

# 启用算法链依赖（Ceres/PCL/Open3D），与 core 保持一致
SFM3_ALG_DEPS = 1

include(../../../common.pri)

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp
