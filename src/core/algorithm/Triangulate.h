#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <QObject>
#include <QVector>
#include <array>
#include <tuple>
#include <set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "data/SfmTypes.h"
#include "Initialization.h"

/**
 * @brief 三角测量类 —— 实现增量式SFM（从运动恢复结构）稀疏重建
 *
 * 该类负责从图像匹配结果中恢复相机位姿和三维点云，采用增量式重建流程：
 * 1. 根据 Initialization 提供的初始化对进行三角测量，生成初始三维点云
 * 2. 逐对三角测量 + PnP后方交会注册新相机
 * 3. 每注册一台新相机后调用 BundleAdjustment::bundleAdjustment 局部BA
 * 4. 全部注册完成后调用 BundleAdjustment::globalBundleAdjustment 全局BA+自标定
 *
 * 初始化阶段（种子对选取、相对定向、位姿恢复）已剥离至 Initialization 类。
 */
class Triangulate : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 获取三角测量单例实例
     * @return Triangulate实例指针
     */
    static Triangulate *instance();

    /**
     * @brief 执行增量式三角测量重建（主入口）
     *
     * 完整流程：种子对三角测量 → 贪心选择下一张图像 →
     *          三视图约束三角测量 → PnP注册新相机 → 局部BA → 循环
     *          最终全局BA+自标定精化
     * @param matchPairs 图像对匹配结果列表
     * @param features 图像特征点列表
     * @param initResult 初始化结果（由 Initialization::findInitialPair 提供）
     * @param viewGraph 视图图（用于贪心选择下一张图像）
     * @param outPoints3D 输出：重建的三维点云
     * @param outCameraPoses 输出：估计的相机位姿列表
     * @param fixIntrinsics 是否固定内参（全局BA不优化内参）
     * @return 重建是否成功
     */
    bool triangulate(const std::vector<MatchPair> &matchPairs,
                     const std::vector<FeatureResult> &features,
                     const InitResult &initResult,
                     const ViewGraph &viewGraph,
                     std::vector<Point3D> &outPoints3D,
                     std::vector<CameraPose> &outCameraPoses,
                     bool fixIntrinsics = false);

signals:
    /**
     * @brief 增量重建过程中实时推送新增点云和全量相机位姿
     * @param points 新增点坐标 (x1,y1,z1,x2,y2,z2,...)
     * @param colors 新增点颜色 (r1,g1,b1,r2,g2,b2,...) 范围0-1
     * @param cameraData 全量相机位姿 (cx,cy,cz, dx,dy,dz 每个相机6个float)
     * @param reset 是否重置（清空旧数据，全量替换）
     */
    void incrementalUpdate(const QVector<float> &points,
                           const QVector<float> &colors,
                           const QVector<float> &cameraData,
                           bool reset = false);

public:
    /**
     * @brief 根据图像尺寸估计相机内参矩阵K
     * @param features 图像特征点列表
     * @return 估计的3x3相机内参矩阵
     */
    cv::Mat estimateK(const std::vector<FeatureResult> &features);

private:
    int m_lastEmittedPointCount = 0;  ///< 上次emit时已发送的点数，用于增量推送

private:
    /**
     * @brief 私有构造函数（单例模式）
     */
    explicit Triangulate(QObject *parent = nullptr);

    /**
     * @brief 析构函数（默认）
     */
    ~Triangulate() override = default;

    /**
     * @brief 禁用拷贝构造函数（单例不可拷贝）
     */
    Triangulate(const Triangulate &) = delete;

    /**
     * @brief 禁用拷贝赋值运算符（单例不可拷贝）
     */
    Triangulate &operator=(const Triangulate &) = delete;

    /**
     * @brief 对一对相机进行三角测量（前向交会），计算三维点
     * @param pair 图像匹配对
     * @param K1 第一台相机内参矩阵
     * @param K2 第二台相机内参矩阵
     * @param R1 第一台相机旋转矩阵
     * @param t1 第一台相机平移向量
     * @param R2 第二台相机旋转矩阵
     * @param t2 第二台相机平移向量
     * @param outWorldPoints 输出：三维世界坐标点列表
     * @param outKeepMask 输出：有效点掩码
     */
    void triangulatePair(const MatchPair &pair,
                         const cv::Mat &K1, const cv::Mat &K2,
                         const cv::Mat &R1, const cv::Mat &t1,
                         const cv::Mat &R2, const cv::Mat &t2,
                         std::vector<cv::Point3d> &outWorldPoints,
                         std::vector<bool> *outKeepMask = nullptr);

    /**
     * @brief 后方交会 —— 使用EPNP+RANSAC求解相机位姿
     * @param K 相机内参矩阵
     * @param kps 图像特征点（2D观测）
     * @param pts3d 对应的三维点（3D坐标）
     * @param rvec 输出：旋转向量
     * @param tvec 输出：平移向量
     * @return 是否成功求解
     */
    bool resection(const cv::Mat &K,
                   const std::vector<cv::KeyPoint> &kps,
                   const std::vector<cv::Point3d> &pts3d,
                   cv::Mat &rvec, cv::Mat &tvec);

    /**
     * @brief 基于特征点在多分辨率网格中的分布计算图像权重得分
     *
     * 将图像划分为 2x2、4x4、8x8 三个网格层级，各层级单格权重分别为
     * 1、2、3。统计每个层级中被特征点占据的网格数量，加权求和作为得分。
     * 得分越高，表示特征点在图像中的空间分布越均匀、覆盖越充分。
     *
     * @param imageId 图像ID
     * @param features 图像特征点列表
     * @return 权重得分
     */
    double computeImageFeatureDistributionScore(int imageId,
                                                const std::vector<FeatureResult> &features) const;

    /**
     * @brief 单例实例指针
     */
    static Triangulate *s_instance;
};

#endif // TRIANGULATE_H