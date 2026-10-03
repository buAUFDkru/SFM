# patchmatch_test.pro —— PatchMatch 代价函数优化的正确性回归测试（控制台程序）
#
# 验证 PatchMatchReconstructorV2 中「窗口预计算」优化是数值等价变换：
# 用 golden（优化前实现）与当前实现在随机数据与边界条件下逐位对比。
TEMPLATE = app
CONFIG  += console
CONFIG  -= app_bundle
TARGET   = patchmatch_test

# PatchMatchReconstructorV2.h 经 data/StereoTypes.h 间接依赖 PCL 头文件，
# 因此这里仍需打开算法链依赖门控（PCL/Ceres/Open3D 的 include 与 lib）
SFM3_ALG_DEPS = 1

include(../../../common.pri)

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp
