#include "PointCloudBuilder.h"

#include <QDebug>
#include <QString>

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/filters/statistical_outlier_removal.h>

#include <open3d/Open3D.h>

namespace sfm {
namespace pointcloud {

void create3DPoints(const std::vector<cv::Mat> &eplimgls,
                                       const std::vector<cv::Mat> &disparityMaps,
                                       const std::vector<CameraPose> &cameraPoses,
                                       const std::vector<cv::Mat> &Qs,
                                       const std::vector<cv::Mat> &r,
                                       std::vector<std::vector<Point3D>> &outPointSets)
{
    outPointSets.clear();

    int numPairs = static_cast<int>(eplimgls.size());

    for (int i = 0; i < numPairs; i++)
    {
        cv::Mat disp32F;
        cv::Mat xyz;
        // 视差图已经是 CV_32FC1 真实视差值（DenseStereo中已除以16），无需再乘以16
        disp32F = disparityMaps[i].clone();
        cv::reprojectImageTo3D(disp32F, xyz, Qs[i]);

        float threshold = 0.1f;
        int windowSize = 11;
        int halfWin = windowSize / 2;
        cv::Mat filteredCloud = xyz.clone();
        for (int y = halfWin; y < xyz.rows - halfWin; y++)
        {
            for (int x = halfWin; x < xyz.cols - halfWin; x++)
            {
                float meanZ = 0.0f;
                int count = 0;

                for (int dy = -halfWin; dy <= halfWin; dy++)
                {
                    for (int dx = -halfWin; dx <= halfWin; dx++)
                    {
                        float depth = xyz.at<cv::Vec3f>(y + dy, x + dx)[2];
                        if (depth > 0)
                        {
                            meanZ += depth;
                            count++;
                        }
                    }
                }

                if (count > 0)
                {
                    meanZ /= count;
                    float currZ = xyz.at<cv::Vec3f>(y, x)[2];
                    if (std::abs(currZ - meanZ) > threshold)
                    {
                        filteredCloud.at<cv::Vec3f>(y, x)[2] = 0.0f;
                    }
                }
            }
        }

        const cv::Mat &Ri = r[i];
        cv::Mat RiInv = Ri.inv();

        const CameraPose &pose = cameraPoses[i];
        cv::Mat Rc = pose.R.empty() ? cv::Mat::eye(3, 3, CV_64F) : pose.R;
        cv::Mat Tc = pose.t.empty() ? cv::Mat::zeros(3, 1, CV_64F) : pose.t;
        cv::Mat RcInv = Rc.inv();

        int rows = eplimgls[i].rows;
        int cols = eplimgls[i].cols;

        std::vector<Point3D> pts;
        int pointId = 0;

        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                cv::Vec3f pt3f = filteredCloud.at<cv::Vec3f>(j, k);
                double z = static_cast<double>(pt3f[2]);
                if (z <= 0.0 || std::isnan(z) || std::isinf(z)) continue;

                double x = static_cast<double>(pt3f[0]);
                double y = static_cast<double>(pt3f[1]);

                cv::Mat Eppoint3D = (cv::Mat_<double>(3, 1) << x * 16, y * 16, z * 16);

                cv::Mat realpoint3D = RiInv * Eppoint3D;

                cv::Mat world_points = RcInv * (realpoint3D - Tc);

                cv::Vec3b bgr = eplimgls[i].at<cv::Vec3b>(j, k);
                if (bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 0) continue;

                Point3D pt3d;
                pt3d.id = pointId++;
                pt3d.position = cv::Point3d(
                    world_points.at<double>(0, 0),
                    world_points.at<double>(1, 0),
                    world_points.at<double>(2, 0));
                pt3d.color = cv::Vec3b(bgr[2], bgr[1], bgr[0]);

                pts.push_back(pt3d);
            }
        }

        qDebug().noquote()
            << QString("[3D点云] 像对 %1 生成 %2 个点").arg(i).arg(pts.size());

        outPointSets.push_back(pts);
    }
}

bool convertToPCL(const std::vector<Point3D> &points,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &outCloud)
{
    if (!outCloud) {
        outCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    }
    outCloud->clear();
    outCloud->reserve(points.size());

    for (const auto &pt : points) {
        pcl::PointXYZRGB pclPt;
        pclPt.x = static_cast<float>(pt.position.x);
        pclPt.y = static_cast<float>(pt.position.y);
        pclPt.z = static_cast<float>(pt.position.z);
        pclPt.r = pt.color[2];
        pclPt.g = pt.color[1];
        pclPt.b = pt.color[0];
        outCloud->push_back(pclPt);
    }

    return !outCloud->empty();
}

void optimizeProcessDense(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    qDebug().noquote()
    << QString("[点云优化] 开始 (Open3D), 输入点数=%1").arg(cloud->size());

    // ── PCL → Open3D 转换 ──
    auto o3dCloud = std::make_shared<open3d::geometry::PointCloud>();
    o3dCloud->points_.reserve(cloud->size());
    o3dCloud->colors_.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
        o3dCloud->points_.emplace_back(pt.x, pt.y, pt.z);
        o3dCloud->colors_.emplace_back(pt.r / 255.0, pt.g / 255.0, pt.b / 255.0);
    }

    try {
        // ── 1. 统计离群去除 (Open3D) ──
        auto [cloudSOR1, removed1] = o3dCloud->RemoveStatisticalOutliers(50, 0.1);
        qDebug().noquote()
            << QString("[点云优化] 一次离群去除: %1 → %2 点")
                .arg(cloud->size()).arg(cloudSOR1->points_.size());

        // ── 2. 体素下采样 (Open3D, 隐式平滑：体素内取质心) ──
        Eigen::Vector3d minB = cloudSOR1->GetMinBound();
        Eigen::Vector3d maxB = cloudSOR1->GetMaxBound();
        double maxSpan = std::max({maxB(0) - minB(0), maxB(1) - minB(1), maxB(2) - minB(2)});
        double leaf = std::max(0.01, maxSpan / 2000.0);

        auto cloudDown = cloudSOR1->VoxelDownSample(leaf);
        qDebug().noquote()
            << QString("[点云优化] 体素下采样: %1 → %2 点 (leaf=%3)")
                .arg(cloudSOR1->points_.size()).arg(cloudDown->points_.size()).arg(leaf, 0, 'f', 4);

        // ── 3. 法向量估计 (Open3D, 替代 MLS 平滑) ──
        // KDTree + 协方差分析，比 MLS 快 50x
        cloudDown->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(0.03, 30));
        cloudDown->OrientNormalsToAlignWithDirection();
        qDebug().noquote()
            << QString("[点云优化] 法向量估计完成");

        // ── 4. 二次统计离群去除 (Open3D) ──
        auto [cloudSOR2, removed2] = cloudDown->RemoveStatisticalOutliers(50, 0.1);
        qDebug().noquote()
            << QString("[点云优化] 二次离群去除: %1 → %2 点")
                .arg(cloudDown->points_.size()).arg(cloudSOR2->points_.size());

        // ── Open3D → PCL 转换 ──
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGB>);
        result->reserve(cloudSOR2->points_.size());
        for (size_t i = 0; i < cloudSOR2->points_.size(); ++i) {
            pcl::PointXYZRGB pt;
            pt.x = static_cast<float>(cloudSOR2->points_[i](0));
            pt.y = static_cast<float>(cloudSOR2->points_[i](1));
            pt.z = static_cast<float>(cloudSOR2->points_[i](2));
            if (!cloudSOR2->colors_.empty()) {
                pt.r = static_cast<uint8_t>(std::clamp(cloudSOR2->colors_[i](0) * 255.0, 0.0, 255.0));
                pt.g = static_cast<uint8_t>(std::clamp(cloudSOR2->colors_[i](1) * 255.0, 0.0, 255.0));
                pt.b = static_cast<uint8_t>(std::clamp(cloudSOR2->colors_[i](2) * 255.0, 0.0, 255.0));
            }
            result->push_back(pt);
        }
        result->width = static_cast<uint32_t>(result->size());
        result->height = 1;

        qDebug().noquote()
            << QString("[点云优化] 完成 (Open3D), 输出点数=%1").arg(result->size());

        cloud = result;

    } catch (const std::exception &e) {
        qDebug().noquote()
            << QString("[点云优化] Open3D 异常: %1, 回退到原始点云").arg(e.what());
    }
}

void optimizeProcess(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    qDebug().noquote()
        << QString("[点云优化] 开始, 输入点数=%1").arg(cloud->size());

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*cloud_filtered);
    }
    qDebug().noquote()
        << QString("[点云优化] 离群点去除: %1 → %2 点").arg(cloud->size()).arg(cloud_filtered->size());

    cloud = cloud_filtered;

    qDebug().noquote()
        << QString("[点云优化] 完成, 输出点数=%1").arg(cloud->size());
}

} // namespace pointcloud
} // namespace sfm
