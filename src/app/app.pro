# app.pro —— SFMReconstructor 主程序（P4b: 完整 UI）
TEMPLATE = app
TARGET   = SFMReconstructor
QT      += core gui widgets opengl openglwidgets

# 启用算法链依赖（PCL 点云 / Open3D 优化 / OpenCV），见 common.pri 的 SFM3_ALG_DEPS 门控块
SFM3_ALG_DEPS = 1

include(../../common.pri)

# 相对路径由 qmake 自动解析（相对本 .pro 所在目录 src/app → src/core）
INCLUDEPATH += ../core

# 依赖 core 静态库（DESTDIR = $$OUT_PWD）
LIBS += -L$$OUT_PWD/../core -lsfmcore

RESOURCES += resources.qrc

HEADERS += \
    ui/MainWindow.h \
    ui/ConfigPanel.h \
    ui/ControlButtons.h \
    ui/DenseReconstructionDialog.h \
    ui/NewProjectDialog.h \
    ui/SparseReconstructionDialog.h \
    ui/LogWidget/LogConsole.h \
    ui/ParamWidget/ParamFormGenerator.h \
    ui/ParamWidget/AlgorithmComboPanel.h \
    ui/ViewWidget/PointCloudWidget.h

SOURCES += \
    main.cpp \
    ui/MainWindow.cpp \
    ui/ConfigPanel.cpp \
    ui/ControlButtons.cpp \
    ui/DenseReconstructionDialog.cpp \
    ui/NewProjectDialog.cpp \
    ui/SparseReconstructionDialog.cpp \
    ui/LogWidget/LogConsole.cpp \
    ui/ParamWidget/ParamFormGenerator.cpp \
    ui/ParamWidget/AlgorithmComboPanel.cpp \
    ui/ViewWidget/PointCloudWidget.cpp