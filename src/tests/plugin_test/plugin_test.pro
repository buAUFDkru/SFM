# plugin_test.pro —— 插件框架验证测试（控制台程序）
TEMPLATE = app
CONFIG  += console
CONFIG  -= app_bundle
TARGET   = plugin_test

# 启用算法链依赖：测试覆盖单测/编排/重建管线（Ceres/PCL/Open3D）
SFM3_ALG_DEPS = 1

include(../../../common.pri)

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp