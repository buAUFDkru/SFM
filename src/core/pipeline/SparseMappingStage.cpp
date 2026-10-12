#include "pipeline/SparseMappingStage.h"

#include "algorithm/IncrementalMapper.h"
#include "algorithm/Initialization.h"
#include "algorithm/IntrinsicsResolver.h"
#include "algorithm/MatcherService.h"
#include "algorithm/PointCloudBuilder.h"
#include "algorithm/SfmGeometry.h"
#include "algorithm/Triangulate.h"
#include "frame/Coordinate.h"
#include "pipeline/ProjectRepository.h"
#include "pipeline/ReconstructionPipeline.h"
#include "plugin/PluginRegistry.h"

#include <QElapsedTimer>

namespace sfm {

SparseMappingInput SparseMappingStage::fromConfig(const ReconstructionConfig &cfg,
                                                  std::vector<FeatureResult> features,
                                                  const QString &sparsePlyPath)
{
    SparseMappingInput in;
    in.imageFiles = cfg.imageFiles;
    in.features = std::move(features);
    in.matcherType = cfg.matcherType;
    in.matcherParams = cfg.matcherParams;
    in.externalCameraMatrix = cfg.cameraMatrix;
    in.useExifFocal = cfg.useExifFocal;
    in.fixCameraMatrix = cfg.fixCameraMatrix;
    in.refineDistortion = cfg.refineDistortion;
    in.refineFocalPerAxis = cfg.refineFocalPerAxis;
    in.sparsePlyPath = sparsePlyPath;
    return in;
}

SparseMappingResult SparseMappingStage::run(SparseMappingInput in,
                                            Triangulate &triangulate,
                                            ThreadPool *threadPool,
                                            CancellationToken *cancel,
                                            IReconstructionCallbacks *callbacks,
                                            const StageLogFn &log,
                                            const StageProgressFn &progress)
{
    SparseMappingResult out;
    out.features = std::move(in.features);

    auto L = [&](const QString &m) { if (log) log(m); };
    auto P = [&](const QString &s, int pct) { if (progress) progress(s, pct); };
    auto stopped = [&]() {
        return cancel && cancel->isStopRequested();
    };

    QElapsedTimer stageTimer;

    // ── 内参（匹配前确定，便于已知 K 估 E）──
    const IntrinsicsResolveResult intrinsics =
        IntrinsicsResolver::resolve(in.externalCameraMatrix, in.useExifFocal,
                                    in.imageFiles, out.features);
    for (const QString &msg : intrinsics.messages)
        L(msg);
    out.K = intrinsics.K;
    const SfmExif::IntrinsicsGuess &exifInfo = intrinsics.exif;

    // ── Stage 2：匹配 ──
    L(QStringLiteral("-------- [阶段 2/6] 特征匹配 开始 --------"));
    P(QStringLiteral("特征匹配"), 0);
    stageTimer.start();

    MatcherService matcherService(&PluginRegistry::instance());
    if (!matcherService.matchFeatures(
            out.features, in.matcherType, in.matcherParams, out.matchPairs,
            threadPool, log, out.K)) {
        L(QStringLiteral("错误: 特征匹配失败"));
        return out;
    }
    if (stopped()) {
        out.cancelled = true;
        L(QStringLiteral("[特征匹配] 收到停止请求"));
        return out;
    }
    P(QStringLiteral("特征匹配"), 100);
    L(QStringLiteral("[特征匹配] 总计: %1 对图像连接 (%2 s)")
          .arg(out.matchPairs.size())
          .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    L(QStringLiteral("-------- [阶段 2/6] 特征匹配 完成 --------"));

    // ── 连通分量 ──
    int mainCompSize = static_cast<int>(out.features.size());
    {
        const SfmGeom::LargestComponentResult comp =
            SfmGeom::keepLargestComponent(out.matchPairs,
                                          static_cast<int>(out.features.size()));
        mainCompSize = comp.mainSize;
        if (comp.numComponents > 1) {
            L(QStringLiteral("[视图图] 匹配图不连通: %1 个分量, 最大分量 %2/%3 张图, "
                             "剔除跨分量匹配对 %4 条, 孤立图像 %5 张")
                  .arg(comp.numComponents).arg(comp.mainSize)
                  .arg(out.features.size())
                  .arg(static_cast<qlonglong>(comp.removedPairs))
                  .arg(comp.isolatedImages));
            L(QStringLiteral("[视图图] 仅在最大连通分量上重建（小分量无法关联到主模型坐标系）"));
        }
    }

    const int initMaxTrials = 3;
    Initialization initialization;
    const std::vector<InitResult> initCandidates =
        initialization.findInitialPairCandidates(
            out.features, out.matchPairs, out.K, initMaxTrials);
    if (initCandidates.empty()) {
        L(QStringLiteral("错误: 未找到合适的初始化图像对，终止重建"));
        return out;
    }
    L(QStringLiteral("[初始化] 候选种子对: %1 组（将逐个尝试）")
          .arg(initCandidates.size()));

    ViewGraph viewGraph =
        ViewGraph::buildFromPairs(out.matchPairs,
                                  static_cast<int>(out.features.size()));

    // ── Stage 3：增量稀疏 ──
    L(QStringLiteral("-------- [阶段 3/6] 稀疏重建 开始 --------"));
    P(QStringLiteral("稀疏重建"), 0);
    stageTimer.restart();

    IncrementalMapperConfig mapCfg;
    mapCfg.triOptions = triangulate.options();
    mapCfg.triOptions.refineDistortion = in.refineDistortion;
    mapCfg.triOptions.refineFocalPerAxis = in.refineFocalPerAxis;
    mapCfg.triOptions.intrinsicsTrusted = exifInfo.valid;
    mapCfg.fixIntrinsics = in.fixCameraMatrix;
    mapCfg.acceptRegCount = mainCompSize;
    L(QStringLiteral("[相机模型] 畸变精化=%1, fx≠fy=%2, EXIF焦距初值=%3")
          .arg(mapCfg.triOptions.refineDistortion ? QStringLiteral("开")
                                                  : QStringLiteral("关"))
          .arg(mapCfg.triOptions.refineFocalPerAxis ? QStringLiteral("开")
                                                    : QStringLiteral("关"))
          .arg(mapCfg.triOptions.intrinsicsTrusted
                   ? QStringLiteral("是（内参已锁）")
                   : QStringLiteral("否（锁内参禁用）")));

    const IncrementalMapperResult mapResult = IncrementalMapper::run(
        triangulate, initCandidates, out.matchPairs, out.features, viewGraph,
        out.K, mapCfg, cancel, log);

    if (mapResult.cancelled) {
        out.cancelled = true;
        return out;
    }
    if (!mapResult.ok)
        return out;

    std::vector<Point3D> sparsePoints = mapResult.points;
    out.cameraPoses = mapResult.poses;

    if (stopped()) {
        out.cancelled = true;
        L(QStringLiteral("[稀疏重建] 收到停止请求"));
        return out;
    }
    P(QStringLiteral("稀疏重建"), 100);
    L(QStringLiteral("[稀疏重建] 生成: %1 个三维点, %2 个相机位姿 (%3 s)")
          .arg(sparsePoints.size()).arg(out.cameraPoses.size())
          .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    L(QStringLiteral("-------- [阶段 3/6] 稀疏重建 完成 --------"));

    // ── Stage 4：点云优化 ──
    L(QStringLiteral("-------- [阶段 4/6] 点云优化 开始 --------"));
    P(QStringLiteral("点云优化"), 0);
    stageTimer.restart();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    pointcloud::convertToPCL(sparsePoints, cloud);
    pointcloud::optimizeProcess(cloud);

    out.optPoints.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
        Point3D p;
        p.id = static_cast<int>(out.optPoints.size());
        p.position = cv::Point3d(pt.x, pt.y, pt.z);
        p.color = cv::Vec3b(pt.b, pt.g, pt.r);
        out.optPoints.push_back(p);
    }
    P(QStringLiteral("点云优化"), 100);
    L(QStringLiteral("[点云优化] 完成: %1 个点 (%2 s)")
          .arg(out.optPoints.size())
          .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    L(QStringLiteral("-------- [阶段 4/6] 点云优化 完成 --------"));

    // 落盘算法系 + UI 显示系
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sparseAlg(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        pointcloud::convertToPCL(out.optPoints, sparseAlg);
        if (!in.sparsePlyPath.isEmpty()) {
            ProjectRepository::saveCloudPly(sparseAlg, in.sparsePlyPath,
                                            QStringLiteral("稀疏重建"), log);
        }

        out.sparseDisplay.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        *out.sparseDisplay = *sparseAlg;
        frame::toDisplayCloudInPlace(*out.sparseDisplay);
        emitStageCloud(callbacks, out.sparseDisplay, QStringLiteral("sparse"));
    }

    out.ok = true;
    return out;
}

} // namespace sfm
