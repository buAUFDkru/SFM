#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <QObject>
#include <QVector>
#include <array>
#include <functional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "data/SfmTypes.h"
#include "Initialization.h"

/**
 * @brief 稀疏重建（三角测量 / 增量注册）可调参数
 *
 * 阈值默认取自 COLMAP：
 * - incremental_triangulator.h: min_angle=1.5, create_max_angle_error=2.0
 * - incremental_mapper.h: abs_pose_min_num_inliers=30,
 *   abs_pose_min_inlier_ratio=0.25, abs_pose_max_error=12.0, max_reg_trials=3
 */
struct TriangulateOptions
{
    // ── 三角化过滤 ──────────────────────────────────────────────
    double minTriAngleDeg   = 1.5;  ///< 最小三角化角度（度）；角度过小则深度不可靠
    double maxReprojErrorPx = 4.0;  ///< 三角化后允许的最大重投影误差（像素）

    // ── PnP（绝对位姿）校验 ─────────────────────────────────────
    int    absPoseMinNumInliers   = 30;   ///< 最少 2D-3D 内点数
    double absPoseMinInlierRatio  = 0.25; ///< 最小内点率
    double absPoseMaxErrorPx      = 12.0; ///< PnP RANSAC 重投影阈值（像素）
    double absPoseMaxMedianErrPx  = 12.0; ///< PnP 后中位重投影误差上限（像素）

    // ── 增量注册 ────────────────────────────────────────────────
    int minCommonPoints = 8;  ///< 三视图共有点下限；不足时跳过本轮新点三角化，
                              ///< 但**不再终止重建**（可复用历史已三角化点做 PnP）
    int maxRegTrials    = 3;  ///< 同一条 (待注册图像, 已注册图像) 边的最大尝试次数
    int topKCandidates  = 4;  ///< 每轮参与排序的跨边界候选边数量
};

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
 *
 * ### 关键点索引 → 三维点映射（kpTo3D）
 *
 * 重建过程中维护 kpTo3D[imageId][keypointIdx] = point3DIdx，用于：
 * - 统计某张未注册图像"当前能看到多少个已三角化点"（COLMAP NumVisiblePoints3D）
 * - 注册新图像时跨**所有**已注册图像收集 2D-3D 对应，而不是只用本轮新算的点
 *
 * 该映射是阶段二（完整 Track / ObservationManager）的雏形，此处先以最小实现落地，
 * 用于消除"三视图共有点不足即终止重建"的致命缺陷。
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

    /// 设置稀疏重建可调参数
    void setOptions(const TriangulateOptions &options) { m_options = options; }

    /// 获取当前稀疏重建可调参数
    const TriangulateOptions &options() const { return m_options; }

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
     * @param externalK 外部提供的内参矩阵；为空时按图像尺寸自动估计。
     *        此前实现无条件重新估计 K，导致调用方传入的已知内参（例如从
     *        cameramatrix.txt 读入、或 EXIF 焦距推导）在三角测量阶段被丢弃，
     *        与初始化阶段使用的 K 不一致。
     * @return 重建是否成功
     */
    bool triangulate(const std::vector<MatchPair> &matchPairs,
                     const std::vector<FeatureResult> &features,
                     const InitResult &initResult,
                     const ViewGraph &viewGraph,
                     std::vector<Point3D> &outPoints3D,
                     std::vector<CameraPose> &outCameraPoses,
                     bool fixIntrinsics = false,
                     const cv::Mat &externalK = cv::Mat());

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
     * @brief 后方交会 —— RANSAC 求解相机位姿并做质量校验
     *
     * 流程：
     * 1. cv::solvePnPRansac（ITERATIVE，失败回退 EPNP）得到粗位姿与内点
     * 2. 用内点集合调用 cv::solvePnP 精化位姿
     * 3. 质量校验：内点数 ≥ minNumInliers、内点率 ≥ minInlierRatio、
     *    中位重投影误差 ≤ maxReprojErrorPx（COLMAP 对应 abs_pose_* 三个阈值）
     *
     * @param K 相机内参矩阵
     * @param imgPts 图像观测点（2D，像素坐标）
     * @param pts3d 对应的三维点（世界坐标），与 imgPts 一一对应
     * @param rvec 输出：旋转向量（3×1）
     * @param tvec 输出：平移向量（3×1）
     * @param outInlierMask 可选输出：内点掩码（与输入点一一对应，1=内点）
     * @param minNumInliers 最小内点数
     * @param minInlierRatio 最小内点率
     * @param maxReprojErrorPx RANSAC 阈值与中位误差上限（像素）
     * @return 是否成功求解并通过质量校验
     */
    bool resection(const cv::Mat &K,
                   const std::vector<cv::Point2d> &imgPts,
                   const std::vector<cv::Point3d> &pts3d,
                   cv::Mat &rvec, cv::Mat &tvec,
                   std::vector<uchar> *outInlierMask = nullptr,
                   int minNumInliers = 30,
                   double minInlierRatio = 0.25,
                   double maxReprojErrorPx = 12.0);

    /**
     * @brief 统计某张图像当前"能看到"多少个已三角化的三维点
     *
     * 对应 COLMAP ObservationManager::NumVisiblePoints3D：遍历该图像与所有已注册
     * 图像之间的匹配，只要匹配到的关键点已关联三维点即计入（按三维点去重）。
     *
     * 该数量同时决定了：
     * - 该图像能否被注册（需 ≥ absPoseMinNumInliers）
     * - 该图像在候选队列中的优先级（数量越多越优先，几何约束越强）
     *
     * @param imageId 待查询图像ID
     * @param kpTo3D 关键点索引 → 三维点索引映射（按图像下标）
     * @param registeredCameras 已注册相机集合
     * @param findMatchPair 匹配对查找函数
     * @return 可见三维点数量
     */
    int numVisiblePoints3D(
        int imageId,
        const std::vector<std::unordered_map<int, int>> &kpTo3D,
        const std::set<int> &registeredCameras,
        const std::function<const MatchPair *(int, int)> &findMatchPair) const;

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

    /// 稀疏重建可调参数（阈值集中于此，便于按数据集调优）
    TriangulateOptions m_options;
};

#endif // TRIANGULATE_H