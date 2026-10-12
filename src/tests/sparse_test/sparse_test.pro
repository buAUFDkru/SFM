# sparse_test —— 稀疏重建健壮性回归（QtTest）
TEMPLATE = app
CONFIG  += testcase console
CONFIG  -= app_bundle
TARGET   = sparse_test

# 需要算法链依赖（Ceres/PCL/Open3D），与 core 链接一致
SFM3_ALG_DEPS = 1

include(../../../common.pri)

QT += testlib core

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp
