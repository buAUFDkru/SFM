# bench_compare —— 阶段 0：自比层指标 JSON 读写与相对基线漂移告警（QtTest）
#
# 不跑重建，只秒级读 JSON 判定；真实数据集巡跑由 recon_cli 完成。

TEMPLATE = app
CONFIG  += testcase console
CONFIG  -= app_bundle
TARGET   = bench_compare

include(../../../common.pri)

QT += testlib core

INCLUDEPATH += ../common

DEFINES += BENCH_FIXTURES_DIR=\\\"$$PWD/fixtures\\\"

SOURCES += main.cpp
