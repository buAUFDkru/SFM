#ifndef STAGECALLBACKS_H
#define STAGECALLBACKS_H

/**
 * @file StageCallbacks.h
 * @brief 流水线阶段共用的日志 / 进度 / 点云预览回调（F13 编排拆分）
 */

#include <functional>

#include <QString>
#include <QVector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class IReconstructionCallbacks;

namespace sfm {

using StageLogFn = std::function<void(const QString &)>;
using StageProgressFn = std::function<void(const QString &, int)>;

/// 将点云下采样后通过回调发给 UI（显示坐标系）
void emitStageCloud(IReconstructionCallbacks *callbacks,
                    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                    const QString &stageName);

} // namespace sfm

#endif // STAGECALLBACKS_H
