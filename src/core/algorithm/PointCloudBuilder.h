#ifndef POINTCLOUDBUILDER_H
#define POINTCLOUDBUILDER_H

/**
 * @file PointCloudBuilder.h
 * @brief 点云构建 / 格式转换 / 优化（自由函数，ADR-003）
 *
 * 原 PointCloudBuilder 单例无跨调用状态，已改为 sfm::pointcloud 命名空间下的自由函数。
 */

#include <vector>

#include <opencv2/core.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "data/SfmTypes.h"

namespace sfm {
namespace pointcloud {

/**
 * @brief 将稠密视差图转化为真实世界的三维点云
 */
void create3DPoints(const std::vector<cv::Mat> &eplimgls,
                    const std::vector<cv::Mat> &disparityMaps,
                    const std::vector<CameraPose> &cameraPoses,
                    const std::vector<cv::Mat> &Qs,
                    const std::vector<cv::Mat> &r,
                    std::vector<std::vector<Point3D>> &outPointSets);

/**
 * @brief 将内部点格式转换为 PCL 点云（BGR→RGB）
 */
bool convertToPCL(const std::vector<Point3D> &points,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &outCloud);

/**
 * @brief 密集点云优化（Open3D：统计离群 + 体素 + 双边）
 */
void optimizeProcessDense(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

/**
 * @brief 稀疏点云优化（PCL：统计离群 + 体素 + MLS）
 */
void optimizeProcess(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

} // namespace pointcloud
} // namespace sfm

#endif // POINTCLOUDBUILDER_H
