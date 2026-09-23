# app.pro —— SFMReconstructor 主程序
TEMPLATE = app
TARGET   = SFMReconstructor
QT      += widgets

include(../../common.pri)

INCLUDEPATH += ../../core

LIBS += -L$$OUT_PWD/../core -lsfmcore

HEADERS += \
    mainwindow.h

SOURCES += \
    main.cpp \
    mainwindow.cpp