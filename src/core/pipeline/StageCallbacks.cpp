#include "pipeline/StageCallbacks.h"

#include "pipeline/ReconstructionPipeline.h"

namespace sfm {

void emitStageCloud(IReconstructionCallbacks *callbacks,
                    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                    const QString &stageName)
{
    if (!cloud || cloud->empty() || !callbacks) return;
    QVector<float> pts, cls;
    const size_t count = cloud->size();
    const size_t step = (count > 2000000) ? count / 2000000 : 1;
    pts.reserve(static_cast<int>((count / step) * 3));
    cls.reserve(static_cast<int>((count / step) * 3));
    for (size_t i = 0; i < count; i += step) {
        const auto &pt = cloud->points[i];
        pts << pt.x << pt.y << pt.z;
        cls << pt.r / 255.0f << pt.g / 255.0f << pt.b / 255.0f;
    }
    callbacks->onStagePointCloudReady(pts, cls, stageName);
}

} // namespace sfm
