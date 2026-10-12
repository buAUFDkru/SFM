#ifndef SPARSEMAPPINGSTAGE_H
#define SPARSEMAPPINGSTAGE_H

/**
 * @file SparseMappingStage.h
 * @brief 阶段 2–4：匹配 / 连通分量 / 增量重建 / 点云优化（F13）
 *
 * 前置：特征已由 FeatureExtractionStage 填好。
 */

#include <vector>

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "algorithm/CancellationToken.h"
#include "data/SfmTypes.h"
#include "pipeline/StageCallbacks.h"

class Triangulate;
class IReconstructionCallbacks;
struct ReconstructionConfig;
namespace sfm { class ThreadPool; }

namespace sfm {

struct SparseMappingInput {
    QStringList imageFiles;
    std::vector<FeatureResult> features; ///< 调用方移入
    QString matcherType;
    QJsonObject matcherParams;
    cv::Mat externalCameraMatrix; ///< 空则走 IntrinsicsResolver
    bool useExifFocal = true;
    bool fixCameraMatrix = false;
    bool refineDistortion = false;
    bool refineFocalPerAxis = false;
    QString sparsePlyPath; ///< 空则不落盘
};

struct SparseMappingResult {
    bool ok = false;
    bool cancelled = false;
    std::vector<FeatureResult> features;
    std::vector<MatchPair> matchPairs;
    std::vector<CameraPose> cameraPoses;
    std::vector<Point3D> optPoints;
    cv::Mat K;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sparseDisplay; ///< 显示系，供 UI
};

class SparseMappingStage
{
public:
    /// 从管线配置填充输入（features 由调用方移入）
    static SparseMappingInput fromConfig(const ReconstructionConfig &cfg,
                                         std::vector<FeatureResult> features,
                                         const QString &sparsePlyPath);

    static SparseMappingResult run(SparseMappingInput in,
                                   Triangulate &triangulate,
                                   ThreadPool *threadPool,
                                   CancellationToken *cancel,
                                   IReconstructionCallbacks *callbacks,
                                   const StageLogFn &log,
                                   const StageProgressFn &progress);
};

} // namespace sfm

#endif // SPARSEMAPPINGSTAGE_H
