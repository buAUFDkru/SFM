#include "SceneViewMapper.h"

#include "viewmodel/FrustumBuilder.h"

namespace sfm {
namespace view {

PointCloudView toPointCloudView(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                std::size_t maxPoints)
{
    PointCloudView view;
    if (!cloud || cloud->empty())
        return view;

    const std::size_t n = cloud->size();
    const std::size_t step =
        (maxPoints > 0 && n > maxPoints) ? (n / maxPoints) : 1;
    view.positions.reserve((n / step) * 3);
    view.colors.reserve((n / step) * 3);
    for (std::size_t i = 0; i < n; i += step) {
        const auto &pt = cloud->points[i];
        view.positions.push_back(pt.x);
        view.positions.push_back(pt.y);
        view.positions.push_back(pt.z);
        view.colors.push_back(pt.r / 255.0f);
        view.colors.push_back(pt.g / 255.0f);
        view.colors.push_back(pt.b / 255.0f);
    }
    return view;
}

ReconstructionView toReconstructionView(const ReconstructionOutput &output,
                                         const std::vector<std::string> &imageFiles,
                                         bool buildFrusta,
                                         std::size_t maxPoints)
{
    ReconstructionView view;
    view.sparse = toPointCloudView(output.sparseCloud, maxPoints);
    view.dense = toPointCloudView(output.denseCloud, maxPoints);
    view.stats.sparsePoints = static_cast<int>(
        output.sparseCloud ? output.sparseCloud->size() : 0);
    view.stats.densePoints = static_cast<int>(
        output.denseCloud ? output.denseCloud->size() : 0);
    view.stats.reprojectionError = output.reprojectionError;
    view.stats.totalElapsedMs = output.totalElapsedMs;
    int reg = 0;
    for (const auto &p : output.cameraPoses) {
        if (p.registered) ++reg;
    }
    view.stats.registeredCameras = reg;

    if (buildFrusta) {
        const float depth = estimateFrustumDepth(view.sparse);
        view.frusta = buildCameraFrusta(output.cameraPoses, depth, imageFiles);
    }
    return view;
}

} // namespace view
} // namespace sfm
