# patchmatch_test —— PatchMatch 代价函数优化的正确性回归（QtTest）
#
# 验证 PatchMatchReconstructorV2 中「窗口预计算」优化是数值等价变换：
# 与 golden（优化前实现）在随机数据与边界条件下逐位比对。

TEMPLATE = app
CONFIG  += testcase console
CONFIG  -= app_bundle
TARGET   = patchmatch_test

# PatchMatchReconstructorV2.h / data/StereoTypes.h 直接包含 PCL 头文件，
# 因此打开算法链依赖门控（PCL/Ceres/Open3D 的 include 与 lib）。
SFM3_ALG_DEPS = 1

include(../../../common.pri)

QT += testlib core

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp
