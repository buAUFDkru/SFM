# core.pro —— 核心静态库（插件接口 + 插件注册表 + 共享算法 + 重建管线）
TEMPLATE = lib
CONFIG  += staticlib
TARGET   = sfmcore
DESTDIR  = $$OUT_PWD

# 启用算法链依赖（Ceres/PCL/Open3D），见 common.pri 的 SFM3_ALG_DEPS 门控块
SFM3_ALG_DEPS = 1

include(../../common.pri)

INCLUDEPATH += . plugin data algorithm pipeline

HEADERS += \
    plugin/ParamSchema.h \
    plugin/PluginTypes.h \
    plugin/ISfmPlugin.h \
    plugin/IDescriptorPlugin.h \
    plugin/IMatcherPlugin.h \
    plugin/PluginInfo.h \
    plugin/PluginRegistry.h \
    data/SfmTypes.h \
    data/ReconstructionOutput.h \
    data/StereoTypes.h \
    algorithm/ThreadPool.h \
    algorithm/SfmLogger.h \
    algorithm/Initialization.h \
    algorithm/MatcherService.h \
    algorithm/BundleAdjustment.h \
    algorithm/PointCloudBuilder.h \
    algorithm/Triangulate.h \
    algorithm/StereoRectifier.h \
    pipeline/IDenseReconstructor.h \
    pipeline/DenseReconstructionFactory.h \
    pipeline/StereoSGBMReconstructor.h \
    pipeline/PatchMatchReconstructor.h \
    pipeline/ReconstructionPipeline.h \
    pipeline/ReconstructionWorker.h \
    pipeline/PipelineRunner.h

SOURCES += \
    plugin/PluginRegistry.cpp \
    algorithm/ThreadPool.cpp \
    algorithm/SfmLogger.cpp \
    algorithm/Initialization.cpp \
    algorithm/MatcherService.cpp \
    algorithm/BundleAdjustment.cpp \
    algorithm/PointCloudBuilder.cpp \
    algorithm/Triangulate.cpp \
    algorithm/StereoRectifier.cpp \
    pipeline/DenseReconstructionFactory.cpp \
    pipeline/StereoSGBMReconstructor.cpp \
    pipeline/PatchMatchReconstructor.cpp \
    pipeline/ReconstructionPipeline.cpp \
    pipeline/ReconstructionWorker.cpp \
    pipeline/PipelineRunner.cpp