#ifndef RECONSTRUCTIONOUTPUT_H
#define RECONSTRUCTIONOUTPUT_H

#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "data/SfmTypes.h"

/**
 * @brief 输出点云结果结构（算法/主管线层，PCL 依赖）
 *
 * 存储重建的最终输出，包括稀疏点云、稠密点云、相机位姿和统计信息。
 * 依赖 PCL，因此本头文件仅在启用 SFM3_ALG_DEPS 的 TU 中引用。
 */
struct ReconstructionOutput
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sparseCloud;  ///< 稀疏点云（来自三角测量）
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseCloud;   ///< 稠密点云（来自立体匹配）
    std::vector<CameraPose> cameraPoses;                 ///< 所有相机的位姿
    int totalPoints = 0;                                 ///< 点云总点数
    double reprojectionError = 0.0;                     ///< 平均重投影误差（像素）
    double totalElapsedMs = 0.0;                         ///< 总处理耗时（毫秒）

    ReconstructionOutput()
    {
        sparseCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        denseCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    }
};

#endif // RECONSTRUCTIONOUTPUT_H