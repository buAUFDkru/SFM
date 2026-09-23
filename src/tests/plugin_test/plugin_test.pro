# plugin_test.pro —— 插件框架验证测试（控制台程序）
TEMPLATE = app
CONFIG  += console
CONFIG  -= app_bundle
TARGET   = plugin_test

include(../../../common.pri)

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += main.cpp