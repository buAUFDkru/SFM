#include "pipeline/DenseReconstructionStage.h"

#include "algorithm/PointCloudBuilder.h"
#include "frame/Coordinate.h"
#include "pipeline/DenseReconstructionFactory.h"
#include "pipeline/IDenseReconstructor.h"
#include "pipeline/ProjectRepository.h"
#include "pipeline/ReconstructionPipeline.h"

#include <QElapsedTimer>
#include <QFileInfo>

namespace sfm {

DenseStageInput DenseReconstructionStage::fromState(
    const ReconstructionConfig &cfg,
    const QString &densePlyPath,
    const QString &cameraMatrixPath,
    const std::vector<CameraPose> &poses,
    const std::vector<MatchPair> &pairs,
    const std::vector<FeatureResult> &features,
    const cv::Mat &K,
    const std::vector<Point3D> &optPoints,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &existingSparse)
{
    DenseStageInput in;
    in.denseMethod = cfg.denseReconstructionMethod;
    in.denseParams = cfg.denseReconstructionParams;
    in.imageFiles = cfg.imageFiles;
    in.projectPath = cfg.projectPath;
    in.densePlyPath = densePlyPath;
    in.cameraMatrixPath = cameraMatrixPath;
    in.cameraPoses = poses;
    in.matchPairs = pairs;
    in.features = features;
    in.K = K;
    in.optPoints = optPoints;
    in.existingSparseDisplay = existingSparse;
    return in;
}

bool DenseReconstructionStage::run(DenseStageInput in,
                                   ReconstructionOutput *output,
                                   CancellationToken *cancel,
                                   IReconstructionCallbacks *callbacks,
                                   const DenseStageHooks &hooks,
                                   const StageLogFn &log,
                                   const StageProgressFn &progress)
{
    if (!output) return false;

    auto L = [&](const QString &m) { if (log) log(m); };
    auto P = [&](const QString &s, int pct) { if (progress) progress(s, pct); };
    auto stopped = [&]() {
        return cancel && cancel->isStopRequested();
    };

    QElapsedTimer stageTimer;

    L(QStringLiteral("-------- [阶段 5/6] 密集重建 开始 --------"));
    P(QStringLiteral("密集重建"), 0);
    stageTimer.start();

    QString method = in.denseMethod;
    QMap<QString, QVariant> params = in.denseParams;
    if (method.isEmpty()) {
        L(QStringLiteral("[密集重建] 等待选择密集重建方法..."));
        if (hooks.requestMethodIfEmpty)
            hooks.requestMethodIfEmpty();
        if (hooks.methodName)
            method = hooks.methodName();
        if (hooks.methodParams)
            params = hooks.methodParams();
    }

    if (stopped()) {
        L(QStringLiteral("[密集重建] 收到停止请求"));
        return false;
    }

    std::vector<StereoPairResult> stereoResults;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseCloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    try {
        auto reconstructor =
            DenseReconstructionFactory::instance().create(method);
        if (!reconstructor) {
            L(QStringLiteral("[密集重建] 未找到方法: %1，跳过密集重建（方法实现于 P4）")
                  .arg(method));
        } else {
            reconstructor->setConfig(params);
            reconstructor->setLogCallback(log);
            reconstructor->setCancellationToken(cancel);

            DenseReconstructionInput denseInput;
            denseInput.cameraPoses = in.cameraPoses;
            denseInput.matchPairs = in.matchPairs;
            denseInput.features = in.features;
            denseInput.cameraMatrix = in.K;
            denseInput.imageFiles = in.imageFiles;
            denseInput.projectPath = in.projectPath;

            L(QStringLiteral("[密集重建] 使用方法: %1")
                  .arg(reconstructor->displayName()));
            DenseReconstructionOutput denseOutput =
                reconstructor->reconstruct(denseInput);

            stereoResults = std::move(denseOutput.stereoResults);
            denseCloud = std::move(denseOutput.denseCloud);

            if (!denseOutput.success) {
                L(QStringLiteral("[密集重建] 失败: %1")
                      .arg(denseOutput.errorMessage));
            } else {
                L(QStringLiteral("[密集重建] 内部完成, 密集点云=%1 点")
                      .arg(denseCloud ? static_cast<int>(denseCloud->size())
                                      : 0));
            }
        }

        if (stopped()) {
            L(QStringLiteral("[密集重建] 收到停止请求"));
            return false;
        }
        P(QStringLiteral("密集重建"), 100);

        int validDepthMaps = 0;
        for (const auto &r : stereoResults) {
            if (!r.depth.empty()) ++validDepthMaps;
        }
        L(QStringLiteral("[密集重建] 完成: %1/%2 对生成有效深度图, 密集点云 %3 点 (%4 s)")
              .arg(validDepthMaps).arg(stereoResults.size())
              .arg(denseCloud ? static_cast<int>(denseCloud->size()) : 0)
              .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    } catch (const std::exception &e) {
        L(QStringLiteral("[密集重建] 异常: %1，跳过密集重建").arg(e.what()));
    } catch (...) {
        L(QStringLiteral("[密集重建] 未知异常，跳过密集重建"));
    }
    L(QStringLiteral("-------- [阶段 5/6] 密集重建 完成 --------"));

    L(QStringLiteral("-------- [阶段 6/6] 密集点云生成 开始 --------"));
    P(QStringLiteral("密集点云生成"), 0);
    stageTimer.restart();

    try {
        int totalPoints = 0;
        if (denseCloud && !denseCloud->empty()) {
            totalPoints = static_cast<int>(denseCloud->size());
            pointcloud::optimizeProcessDense(denseCloud);
        }
        L(QStringLiteral("[密集点云] 反投影完成, 总点数=%1").arg(totalPoints));

        if (stopped()) {
            L(QStringLiteral("[密集点云生成] 收到停止请求"));
            return false;
        }
        P(QStringLiteral("密集点云生成"), 100);
        L(QStringLiteral("[密集点云生成] 完成: %1 个点 (%2 s)")
              .arg(denseCloud ? static_cast<int>(denseCloud->size()) : 0)
              .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    } catch (const std::exception &e) {
        L(QStringLiteral("[密集点云生成] 异常: %1").arg(e.what()));
    } catch (...) {
        L(QStringLiteral("[密集点云生成] 未知异常"));
    }
    L(QStringLiteral("-------- [阶段 6/6] 密集点云生成 完成 --------"));

    if (denseCloud && !denseCloud->empty()) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseDisplay(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        denseDisplay->resize(denseCloud->size());
        for (size_t i = 0; i < denseCloud->size(); ++i)
            denseDisplay->points[i] = denseCloud->points[i];
        frame::toDisplayCloudInPlace(*denseDisplay);
        emitStageCloud(callbacks, denseDisplay, QStringLiteral("dense"));
        if (!in.densePlyPath.isEmpty()) {
            ProjectRepository::saveCloudPly(denseCloud, in.densePlyPath,
                                            QStringLiteral("密集重建"), log);
        }
    }

    // 组装输出（显示系）
    if (!in.optPoints.empty()) {
        pointcloud::convertToPCL(in.optPoints, output->sparseCloud);
        frame::toDisplayCloudInPlace(*output->sparseCloud);
    } else if (in.existingSparseDisplay && !in.existingSparseDisplay->empty()) {
        output->sparseCloud = in.existingSparseDisplay;
    }

    if (denseCloud && !denseCloud->empty()) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseOutput(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        denseOutput->resize(denseCloud->size());
        for (size_t i = 0; i < denseCloud->size(); ++i)
            denseOutput->points[i] = denseCloud->points[i];
        frame::toDisplayCloudInPlace(*denseOutput);
        output->denseCloud = denseOutput;
        L(QStringLiteral("[输出] 稀疏点云: %1 点, 密集点云: %2 点")
              .arg(output->sparseCloud
                       ? static_cast<int>(output->sparseCloud->size())
                       : 0)
              .arg(static_cast<int>(output->denseCloud->size())));
    } else {
        L(QStringLiteral("[输出] 稀疏点云: %1 点, 密集点云: 空（密集重建未生成有效点）")
              .arg(output->sparseCloud
                       ? static_cast<int>(output->sparseCloud->size())
                       : 0));
    }

    output->cameraPoses = in.cameraPoses;
    output->totalPoints = !in.optPoints.empty()
                              ? static_cast<int>(in.optPoints.size())
                              : (output->sparseCloud
                                     ? static_cast<int>(output->sparseCloud->size())
                                     : 0);

    if (!in.cameraMatrixPath.isEmpty() && !in.imageFiles.isEmpty()) {
        QString camErr;
        if (ProjectRepository::writeCameraMatrixTxt(
                in.cameraMatrixPath, in.cameraPoses, in.imageFiles, &camErr)) {
            L(QStringLiteral("[相机参数] 已保存到 %1").arg(in.cameraMatrixPath));
        } else if (!camErr.isEmpty()) {
            L(QStringLiteral("[相机参数] %1").arg(camErr));
        }
    }

    L(QStringLiteral("[输出] 稀疏点云: %1 点").arg(in.optPoints.size()));
    return true;
}

} // namespace sfm
