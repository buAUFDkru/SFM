# ui_test.pro —— UI 组件测试（QTest + offscreen Qt 平台）
TEMPLATE = app
CONFIG  += testcase
CONFIG  -= app_bundle
TARGET   = ui_test

# 启用算法链依赖（PCL 点云 / OpenCV），见 common.pri 的 SFM3_ALG_DEPS 门控块
SFM3_ALG_DEPS = 1

include(../../../common.pri)

QT += testlib widgets opengl openglwidgets

# 相对路径由 qmake 自动解析（相对本 .pro 所在目录 src/tests/ui_test）
INCLUDEPATH += ../../core
INCLUDEPATH += ../../app/ui

LIBS += -L$$OUT_PWD/../../core -lsfmcore

SOURCES += \
    main.cpp \
    ../../app/ui/MainWindow.cpp \
    ../../app/ui/ConfigPanel.cpp \
    ../../app/ui/ControlButtons.cpp \
    ../../app/ui/DenseReconstructionDialog.cpp \
    ../../app/ui/LogWidget/LogConsole.cpp \
    ../../app/ui/ParamWidget/ParamFormGenerator.cpp \
    ../../app/ui/ParamWidget/AlgorithmComboPanel.cpp \
    ../../app/ui/ViewWidget/PointCloudWidget.cpp

HEADERS += \
    ../../app/ui/MainWindow.h \
    ../../app/ui/ConfigPanel.h \
    ../../app/ui/ControlButtons.h \
    ../../app/ui/DenseReconstructionDialog.h \
    ../../app/ui/LogWidget/LogConsole.h \
    ../../app/ui/ParamWidget/ParamFormGenerator.h \
    ../../app/ui/ParamWidget/AlgorithmComboPanel.h \
    ../../app/ui/ViewWidget/PointCloudWidget.h

# 插件目录（MainWindow 冒烟测试用），构建时替换为应用的插件部署目录
DEFINES += PLUGINS_DIR_DEFAULT=\\\"$$OUT_PWD/../../app/plugins\\\"