# core.pro —— 核心静态库（插件接口 + 插件注册表 + 共享算法 + 重建管线）
TEMPLATE = lib
CONFIG  += staticlib
TARGET   = sfmcore
DESTDIR  = $$OUT_PWD

# 启用算法链依赖（Ceres/PCL/Open3D），见 common.pri 的 SFM3_ALG_DEPS 门控块
SFM3_ALG_DEPS = 1

include(../../common.pri)

INCLUDEPATH += . plugin data algorithm pipeline builtin frame viewmodel

HEADERS += \
    frame/Coordinate.h \
    viewmodel/RenderModel.h \
    viewmodel/FrustumBuilder.h \
    viewmodel/SceneViewMapper.h \
    algorithm/CancellationToken.h \
    algorithm/ProgressSink.h \
    plugin/ParamSchema.h \
    plugin/ParamSchemaUtils.h \
    plugin/PluginTypes.h \
    plugin/ISfmPlugin.h \
    plugin/IDescriptorPlugin.h \
    plugin/IMatcherPlugin.h \
    plugin/PluginInfo.h \
    plugin/PluginRegistry.h \
    builtin/BuiltinDescriptors.h \
    builtin/BuiltinMatchers.h \
    builtin/registerBuiltIn.h \
    data/SfmTypes.h \
    data/ExifReader.h \
    data/ReconstructionOutput.h \
    data/StereoTypes.h \
    algorithm/ThreadPool.h \
    algorithm/SfmLogger.h \
    algorithm/SfmGeometry.h \
    algorithm/TwoViewGeometry.h \
    algorithm/CorrespondenceGraph.h \
    algorithm/ObservationManager.h \
    algorithm/IncrementalTriangulator.h \
    algorithm/Initialization.h \
    algorithm/MatcherService.h \
    algorithm/BundleAdjustment.h \
    algorithm/PointCloudBuilder.h \
    algorithm/IntrinsicsResolver.h \
    algorithm/IncrementalMapper.h \
    algorithm/Triangulate.h \
    algorithm/dense/PatchMatchTypes.h \
    algorithm/dense/strategies.h \
    algorithm/dense/PatchMatchShared.h \
    algorithm/dense/PatchMatchCore.h \
    algorithm/dense/PatchMatchCoreT.h \
    algorithm/dense/PatchMatchCost.h \
    algorithm/dense/PatchMatchPropagate.h \
    algorithm/dense/PatchMatchRefine.h \
    algorithm/dense/PatchMatchFusion.h \
    pipeline/IDenseReconstructor.h \
    pipeline/DenseReconstructionFactory.h \
    pipeline/PatchMatchReconstructor.h \
    pipeline/PatchMatchReconstructorV2.h \
    pipeline/ReconstructionPipeline.h \
    pipeline/ReconstructionConfigIO.h \
    pipeline/ReconstructionWorker.h \
    pipeline/PipelineRunner.h \
    pipeline/ProjectStore.h \
    pipeline/ProjectRepository.h \
    pipeline/StageCallbacks.h \
    pipeline/FeatureExtractionStage.h \
    pipeline/SparseMappingStage.h \
    pipeline/DenseReconstructionStage.h

SOURCES += \
    plugin/ParamSchemaUtils.cpp \
    plugin/PluginRegistry.cpp \
    data/ExifReader.cpp \
    builtin/BuiltinDescriptors.cpp \
    builtin/BuiltinMatchers.cpp \
    builtin/registerBuiltIn.cpp \
    algorithm/ThreadPool.cpp \
    algorithm/SfmLogger.cpp \
    algorithm/TwoViewGeometry.cpp \
    algorithm/CorrespondenceGraph.cpp \
    algorithm/ObservationManager.cpp \
    algorithm/IncrementalTriangulator.cpp \
    algorithm/Initialization.cpp \
    algorithm/MatcherService.cpp \
    algorithm/BundleAdjustment.cpp \
    algorithm/PointCloudBuilder.cpp \
    algorithm/IntrinsicsResolver.cpp \
    algorithm/IncrementalMapper.cpp \
    algorithm/Triangulate.cpp \
    algorithm/dense/PatchMatchShared.cpp \
    algorithm/dense/PatchMatchCost.cpp \
    algorithm/dense/PatchMatchPropagate.cpp \
    algorithm/dense/PatchMatchRefine.cpp \
    algorithm/dense/PatchMatchFusion.cpp \
    algorithm/dense/strategies.cpp \
    algorithm/dense/PatchMatchCore.cpp \
    pipeline/DenseReconstructionFactory.cpp \
    pipeline/PatchMatchReconstructor.cpp \
    pipeline/PatchMatchReconstructorV2.cpp \
    pipeline/ReconstructionPipeline.cpp \
    pipeline/ReconstructionConfigIO.cpp \
    pipeline/ReconstructionWorker.cpp \
    pipeline/PipelineRunner.cpp \
    pipeline/ProjectStore.cpp \
    pipeline/ProjectRepository.cpp \
    pipeline/StageCallbacks.cpp \
    pipeline/FeatureExtractionStage.cpp \
    pipeline/SparseMappingStage.cpp \
    pipeline/DenseReconstructionStage.cpp \
    viewmodel/FrustumBuilder.cpp \
    viewmodel/SceneViewMapper.cpp

RESOURCES += \
    presets/presets.qrc