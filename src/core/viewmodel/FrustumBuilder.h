#ifndef SFM_VIEWMODEL_FRUSTUMBUILDER_H
#define SFM_VIEWMODEL_FRUSTUMBUILDER_H

/**
 * @file FrustumBuilder.h
 * @brief 相机视锥金字塔几何（从 MainWindow 下沉，可单测）
 */

#include <string>
#include <vector>

#include "data/SfmTypes.h"
#include "viewmodel/RenderModel.h"

namespace sfm {
namespace view {

/**
 * @param poses 算法坐标系位姿（内部转换到显示系）
 * @param depth 金字塔深度（显示系单位，通常取点云包围盒边长的 12%）
 * @param imageFiles 与 imageId 对齐的图像路径（可空）
 */
CameraFrustumView buildCameraFrusta(const std::vector<CameraPose> &poses,
                                    float depth,
                                    const std::vector<std::string> &imageFiles);

/// 仅线框线段（测试便捷）
inline std::vector<float> buildCameraFrustumLines(const std::vector<CameraPose> &poses,
                                                  float depth)
{
    return buildCameraFrusta(poses, depth, {}).lineSegments;
}

/**
 * @brief 由显示系点云包围盒估计金字塔深度（最大边长 × 0.12，下限 1）
 */
float estimateFrustumDepth(const PointCloudView &cloud,
                           float fraction = 0.12f,
                           float fallback = 1.0f);

} // namespace view
} // namespace sfm

#endif // SFM_VIEWMODEL_FRUSTUMBUILDER_H
