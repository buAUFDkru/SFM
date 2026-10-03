# sparse_test.pro —— 稀疏重建（阶段一健壮性改造）回归测试
TEMPLATE = app
CONFIG  += console
CONFIG  -= app_bundle
TARGET   = sparse_test

# 启用算法链依赖（Ceres/PCL/Open3D），与 core 保持一致
SFM3_ALG_DEPS = 1

include(../../../common.pri)

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp
