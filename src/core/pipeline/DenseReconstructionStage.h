#ifndef DENSERECONSTRUCTIONSTAGE_H
#define DENSERECONSTRUCTIONSTAGE_H

/**
 * @file DenseReconstructionStage.h
 * @brief 阶段 5–6：密集重建方法执行 + 点云组装（F13）
 */

#include <functional>
#include <vector>

#include <QMap>
#include <QString>
#include <QStringList>
#include <QVariant>

#include <opencv2/core.hpp>

#include "algorithm/CancellationToken.h"
#include "data/ReconstructionOutput.h"
#include "data/SfmTypes.h"
#include "pipeline/StageCallbacks.h"

class IReconstructionCallbacks;
struct ReconstructionConfig;

namespace sfm {

struct DenseStageInput {
    QString denseMethod;
    QMap<QString, QVariant> denseParams;
    QStringList imageFiles;
    QString projectPath;
    QString densePlyPath;     ///< 空则不落盘
    QString cameraMatrixPath; ///< 空则不写 cameramatrix.txt

    std::vector<CameraPose> cameraPoses;
    std::vector<MatchPair> matchPairs;
    std::vector<FeatureResult> features;
    cv::Mat K;
    std::vector<Point3D> optPoints;
    /// loadProject 后续算：无 optPoints 时沿用已加载的显示系稀疏云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr existingSparseDisplay;
};

struct DenseStageHooks {
    /// 配置中方法名为空时调用，由管线 waitForDenseReconstructionMethod
    std::function<void()> requestMethodIfEmpty;
    /// 请求后重新读取方法名/参数（因 UI 可能已写入 config）
    std::function<QString()> methodName;
    std::function<QMap<QString, QVariant>()> methodParams;
};

class DenseReconstructionStage
{
public:
    static DenseStageInput fromState(const ReconstructionConfig &cfg,
                                     const QString &densePlyPath,
                                     const QString &cameraMatrixPath,
                                     const std::vector<CameraPose> &poses,
                                     const std::vector<MatchPair> &pairs,
                                     const std::vector<FeatureResult> &features,
                                     const cv::Mat &K,
                                     const std::vector<Point3D> &optPoints,
                                     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &existingSparse);

    /// 向 output 写入稀疏/密集显示系点云与相机位姿；取消时返回 false
    static bool run(DenseStageInput in,
                    ReconstructionOutput *output,
                    CancellationToken *cancel,
                    IReconstructionCallbacks *callbacks,
                    const DenseStageHooks &hooks,
                    const StageLogFn &log,
                    const StageProgressFn &progress);
};

} // namespace sfm

#endif // DENSERECONSTRUCTIONSTAGE_H
