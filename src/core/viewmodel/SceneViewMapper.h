#ifndef SFM_VIEWMODEL_SCENEVIEWMAPPER_H
#define SFM_VIEWMODEL_SCENEVIEWMAPPER_H

/**
 * @file SceneViewMapper.h
 * @brief ReconstructionOutput（PCL）→ ReconstructionView（无第三方类型）
 */

#include <cstddef>
#include <string>
#include <vector>

#include "data/ReconstructionOutput.h"
#include "viewmodel/RenderModel.h"

namespace sfm {
namespace view {

/// PCL 点云 → PointCloudView；maxPoints>0 时均匀下采样
PointCloudView toPointCloudView(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                std::size_t maxPoints = 5000000);

/// 组装完整 ReconstructionView（含可选视锥）
ReconstructionView toReconstructionView(const ReconstructionOutput &output,
                                         const std::vector<std::string> &imageFiles = {},
                                         bool buildFrusta = true,
                                         std::size_t maxPoints = 5000000);

} // namespace view
} // namespace sfm

#endif // SFM_VIEWMODEL_SCENEVIEWMAPPER_H
