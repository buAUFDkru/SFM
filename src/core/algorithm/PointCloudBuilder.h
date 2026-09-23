#ifndef POINTCLOUDBUILDER_H
#define POINTCLOUDBUILDER_H

#include <QObject>
#include <QDebug>
#include <vector>

#include <opencv2/core.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "data/SfmTypes.h"

/**
 * @brief 点云构建器类
 *
 * 负责从视差图生成三维点云并进行PCL优化处理。
 * 该类是单例模式，确保整个应用中只有一个实例。
 */
class PointCloudBuilder : public QObject
{
    Q_OBJECT

public:
    static PointCloudBuilder *instance();

    /**
     * @brief 生成3D点云
     *
     * 将稠密视差图转化为真实世界的三维点云。
     * 对每一对立体像对执行：
     * - 利用视差和Q矩阵恢复核线坐标系下的三维坐标
     * - 对点云进行统计滤波（局部窗口深度均值差异过大的点Z置零）
     * - 依次通过核线旋转矩阵和相机外参将点云变换至世界坐标系
     * - 筛选有效点并存入输出列表
     *
     * @param eplimgls 左核线影像
     * @param disparityMaps 视差图
     * @param cameraPoses 相机外参（R,t）
     * @param Qs 深度矩阵
     * @param r 核线到原图的旋转矩阵（R1矩阵列表）
     * @param outPointSets 输出的三维点云（每对一组）
     */
    void create3DPoints(const std::vector<cv::Mat> &eplimgls,
                        const std::vector<cv::Mat> &disparityMaps,
                        const std::vector<CameraPose> &cameraPoses,
                        const std::vector<cv::Mat> &Qs,
                        const std::vector<cv::Mat> &r,
                        std::vector<std::vector<Point3D>> &outPointSets);

    /**
     * @brief 将内部点格式转换为PCL点云
     *
     * 将自定义的std::vector<Point3D>格式转换为PCL的pcl::PointXYZRGB格式，
     * 同时进行颜色空间转换（BGR→RGB）。
     *
     * @param points 输入的自定义格式点云
     * @param outCloud 输出的PCL点云智能指针
     * @return 是否转换成功
     */
    bool convertToPCL(const std::vector<Point3D> &points,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &outCloud);

    /**
     * @brief 密集点云优化处理管线 (Open3D)
     *
     * 使用 Open3D 替代 PCL，处理千万级点云性能提升 50x：
     * 1. 统计离群去除 (RemoveStatisticalOutliers, nb=50, std=0.1)
     * 2. 体素下采样 (VoxelDownSample, 自适应leaf, 隐式平滑=体素质心)
     * 3. 法向量估计 (EstimateNormals, 双边滤波需要)
     * 4. 双边滤波 (KDTree手动实现, 沿法向平滑保边去噪)
     * 5. 二次统计离群去除 (RemoveStatisticalOutliers, nb=50, std=0.1)
     *
     * @param cloud 输入/输出点云 (PCL PointXYZRGB)
     */
    void optimizeProcessDense(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
    /**
     * @brief 点云优化处理管线
     *
     * 执行完整的点云优化流程：
     * 1. 统计离群点去除（meanK=50, stdDevMulThresh=1）
     * 2. 体素下采样（leafSize=0.01）
     * 3. MLS移动最小二乘平滑与上采样
     * 4. 二次统计离群点去除（meanK=50, stdDevMulThresh=0.1）
     *
     * @param cloud 输入/输出点云
     */
    void optimizeProcess(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

private:
    explicit PointCloudBuilder(QObject *parent = nullptr);
    ~PointCloudBuilder() override = default;
    PointCloudBuilder(const PointCloudBuilder &) = delete;
    PointCloudBuilder &operator=(const PointCloudBuilder &) = delete;

    static PointCloudBuilder *s_instance;
};

#endif // POINTCLOUDBUILDER_H
