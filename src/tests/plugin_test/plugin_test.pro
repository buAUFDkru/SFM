# plugin_test —— 插件体系验证（QtTest）
TEMPLATE = app
CONFIG  += testcase console
CONFIG  -= app_bundle
TARGET   = plugin_test

# 需要算法链依赖，覆盖描述/匹配/重建全链路（Ceres/PCL/Open3D）
SFM3_ALG_DEPS = 1

include(../../../common.pri)

QT += testlib core

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp
