#ifndef SFM_VIEWMODEL_RENDERMODEL_H
#define SFM_VIEWMODEL_RENDERMODEL_H

/**
 * @file RenderModel.h
 * @brief 渲染边界 DTO（ADR-001）—— 无 PCL / Ceres / Open3D 依赖
 *
 * UI 与批处理只消费本头文件中的类型；算法侧的 ReconstructionOutput 留在 core 内部。
 * 坐标约定：positions / frusta 均为**显示坐标系**（见 sfm::frame::toDisplay）。
 */

#include <cstddef>
#include <string>
#include <vector>

namespace sfm {
namespace view {

struct PointCloudView {
    std::vector<float> positions; ///< xyz 展平
    std::vector<float> colors;    ///< rgb，0~1
    std::size_t size() const noexcept { return positions.size() / 3; }
    bool empty() const noexcept { return positions.empty(); }
};

struct CameraFrustumView {
    std::vector<float> lineSegments; ///< 每 6 float 一条线段（a.xyz, b.xyz）
    std::vector<float> quadCorners;  ///< 每相机 12 float = 4 角 × xyz
    std::vector<std::string> imagePaths;
    std::size_t cameraCount() const noexcept { return imagePaths.size(); }
};

struct SceneStats {
    int sparsePoints = 0;
    int densePoints = 0;
    int registeredCameras = 0;
    double reprojectionError = 0.0;
    double totalElapsedMs = 0.0;
};

struct ReconstructionView {
    PointCloudView sparse;
    PointCloudView dense;
    CameraFrustumView frusta;
    SceneStats stats;
};

} // namespace view
} // namespace sfm

#endif // SFM_VIEWMODEL_RENDERMODEL_H
