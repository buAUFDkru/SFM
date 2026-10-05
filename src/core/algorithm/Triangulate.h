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

#include <memory>
#include <set>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "data/SfmTypes.h"
#include "Initialization.h"
#include "CorrespondenceGraph.h"
#include "ObservationManager.h"
#include "IncrementalTriangulator.h"

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
    int minCommonPoints = 8;  ///< 三视图共有点下限（阶段一遗留；阶段二已不再依赖它决定成败）
    int maxRegTrials    = 3;  ///< 同一张图像的最大注册尝试次数（COLMAP max_reg_trials）

    // ── 局部BA（阶段二，对应 COLMAP ba_local_*）───────────────────
    int    baLocalNumImages = 6;      ///< 局部BA参与优化的邻域图像数量
    int    baLocalMaxRefinements = 2; ///< 局部BA最大迭代精化次数
    double baLocalMaxRefinementChange = 0.001; ///< 变化率低于此值即停止精化
    int    baLocalMaxTrackLength = 15;///< 轨迹长于此值的点视为已稳定，不参与局部BA
    /// 局部BA的Ceres迭代上限（COLMAP ba_local_max_num_iterations = 25）。
    /// 原为硬编码 100，实测每次跑到 43~61 次迭代，占整个重建 46% 的时间。
    int    baLocalMaxIterations = 25;

    // ── 全局BA与精化 ──────────────────────────────────────────────
    int    globalMaxRefinements = 3;  ///< 末尾全局BA迭代精化次数
    double globalMaxRefinementChange = 0.0005;
    /// 全局BA的Ceres迭代上限（COLMAP ba_global_max_num_iterations = 50）。
    /// 原为硬编码 200；收尾三轮实测只用到 25/16/16 次。
    int    baGlobalMaxIterations = 50;

    // ── 外点过滤 ──────────────────────────────────────────────────
    double filterMaxReprojErrorPx = 4.0;  ///< 过滤重投影误差阈值（像素）
    double filterMinTriAngleDeg   = 1.5;  ///< 过滤三角化角下限（度）

    // ── 阶段四 4.4：末尾滤波细化 ──────────────────────────────────
    /**
     * @brief 末尾滤波的轨迹长度下限
     *
     * 重建收尾时删除观测数少于此值的点：轨迹长度 1 的点只有一个观测，
     * 没有冗余约束、无法检验也无法被 BA 有效约束（COLMAP min_track_len 语义）。
     */
    size_t filterMinTrackLength = 2;

    // ── 阶段四 4.6：场景归一化 ────────────────────────────────────
    bool   enableSceneNormalization = true; ///< 收尾时是否做场景归一化
    /**
     * @brief 归一化目标尺度：已注册相机光心到质心的平均距离
     *
     * COLMAP NormalizeReconstruction 语义。自标定的尺度本是任意的，
     * 归一化把任意尺度固定下来，提升下游（密集深度范围估计、可视化）数值稳定。
     */
    double sceneNormalizationExtent = 10.0;

    /**
     * @brief 种子对初始化后首轮过滤的放宽阈值（像素）
     *
     * 此时内参尚未标定好（尤其是从图像尺寸按 60° FOV 估算的焦距，
     * 与无人机相机真实的 ~84° 相差可达 30%），重投影误差普遍偏大。
     * 用 COLMAP 的 4px 严格阈值会把点全部滤光导致重建失败，
     * 因此首轮放宽，待全局BA自标定收敛后再收紧到 4px。
     */
    double initialFilterMaxReprojErrorPx = 12.0;

    // ── 初始化三角化 ──────────────────────────────────────────────
    /**
     * @brief 种子对三角化角下限（COLMAP init_min_tri_angle = 16°）
     *
     * COLMAP 的 16° 依赖「种子对足够宽基线」这一隐含前提。视角变化平缓的
     * 无人机序列达不到该角度，若死守 16° 会导致种子对一个点都建不出来。
     * 实际执行时按 16° → 8° → 4° → 2° → minTriAngleDeg 阶梯放宽，
     * 直到三角化出 ≥ minSeedPoints3D 个点。
     */
    double initMinTriAngleDeg = 16.0;
    int minSeedPoints3D = 100;  ///< 种子对至少应三角化出的点数，不足则继续放宽门槛

    // ── 阶段五 5.2：径向/切向畸变（全局BA联合精化）────────────────────────
    /**
     * @brief 是否在全局BA中联合精化 k1,k2,p1,p2
     *
     * 注意：本开关**默认关闭**。实机对照（三数据集各 2 轮）显示打开后
     * 重投影中位下降但点云点数腰斩（106MEDIA 6389→2769），代价大于收益；
     * 这里留着开关，强畸变镜头（|k1| > 0.1）或纯精度验证场景再开。
     */
    bool refineDistortion = false;

    /**
     * @brief 精化内参（焦距 / 畸变）所需的已注册相机数下限
     *
     * 只有 2 台相机、几百个点时，焦距与「场景尺度」是一对一的同义变换，
     * 残差下降与焦距减小一样好 —— 自标定会无约束地往任意方向滑。
     * 实测 106MEDIA 种子对阶段 f 在 3228 / 3675 / 4176 之间乱跳（真值 ≈3660），
     * 而收敛到 3228 或 4176 的那几次后续重建全部失败「初始全局BA后无有效三维点」。
     *
     * 因此焦距与畸变一起门控：注册相机数不足时全局 BA 把内参完全钉死，
     * 只用针孔模型精化外参与三维点；到收尾阶段（相机已注册满）再放开。
     */
    int intrinsicsMinCameras = 8;

    /**
     * @brief 初值内参是否可信（由阶段五 5.1 决定）
     *
     * - `true`  —— K 来自 EXIF 焦距反推。这类值已经很接近真值（实测
     *   106MEDIA 3659.7 vs 自标定收敛 3676，偏差 0.4%），种子对阶段
     *   「锁死焦距」只会把真值附近的解钉得更稳。
     * - `false` —— K 来自 `estimateK` 的 60°FOV 猜测，可能偏 20%+。此时
     *   必须沿用旧行为（任何相机数都放开焦距），否则锁死在一个错误焦距上
     *   会直接压垮重建：实测 data（guess 1146 vs 收敛 912）点数从 10334
     *   掉到 8016，test14 重投影中位从 0.21 掉到 0.58px。
     */
    bool intrinsicsTrusted = false;

    /// 5.3：允许 fx ≠ fy（默认关闭 —— 无人机相机像素为正方形）
    bool refineFocalPerAxis = false;

    // ── 阶段二开关 ────────────────────────────────────────────────
    bool enableTrackCompletion  = true; ///< 是否执行 Complete / Merge
    bool enableRetriangulation  = true; ///< 是否执行 Retriangulate
    bool allowTwoViewTracksFallback = true; ///< 种子对无点时是否放宽「忽略两视图轨迹」
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

    /// 设置增量三角化器参数（Complete / Merge / Retriangulate 的阈值）
    void setTriangulatorOptions(const TriangulatorOptions &options)
    {
        m_triOptions = options;
    }

    /// 获取当前增量三角化器参数
    const TriangulatorOptions &triangulatorOptions() const { return m_triOptions; }

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
     * @param dist 可选畸变系数（k1,k2,p1,p2,k3）；空表示针孔模型
     * @return 是否成功求解并通过质量校验
     */
    bool resection(const cv::Mat &K,
                   const std::vector<cv::Point2d> &imgPts,
                   const std::vector<cv::Point3d> &pts3d,
                   cv::Mat &rvec, cv::Mat &tvec,
                   std::vector<uchar> *outInlierMask = nullptr,
                   int minNumInliers = 30,
                   double minInlierRatio = 0.25,
                   double maxReprojErrorPx = 12.0,
                   const cv::Mat &dist = cv::Mat());

    // ══════════════════════════════════════════════════════════════════════
    //  阶段二：增量重建子步骤（对应 COLMAP IncrementalMapper 的同名流程）
    // ══════════════════════════════════════════════════════════════════════

    /**
     * @brief 排名下一批待注册图像（COLMAP FindNextImages）
     *
     * 按「可见三维点数」降序；已尝试次数达上限的图像被排除。
     * @return (可见点数, 图像ID) 列表，已按可见点数降序排列
     */
    std::vector<std::pair<int, int>> findNextImages() const;

    /**
     * @brief 注册一张新图像（COLMAP RegisterNextImage）
     *
     * 1. 收集该图的 2D-3D 对应（来自所有已注册图像的已三角化点）
     * 2. PnP + 内点数 / 内点率 / 中位误差校验
     * 3. 成功后三角化该图上的新点并执行局部BA精化
     * @return 是否注册成功
     */
    bool registerNextImage(int imageId);

    /// 三角化新注册图像 + 迭代局部BA精化
    void triangulateAndRefineImage(int imageId);

    /**
     * @brief 迭代局部BA精化（COLMAP IterativeLocalRefinement）
     *
     * 反复执行「局部BA → 合并轨迹 → 补全轨迹 → 过滤外点」，
     * 直到变化率低于阈值或达到最大次数。
     */
    void iterativeLocalRefinement(int imageId);

    /**
     * @brief 一次局部BA（COLMAP AdjustLocalBundle）
     *
     * 优化「新注册相机 + 共享观测最多的 Top-N 邻域相机」的位姿，
     * 以及本轮被新建/修改的三维点。
     */
    void adjustLocalBundle(int imageId, const std::vector<int> &variablePointIds);

    /**
     * @brief 找出与新注册图像共享三维点最多的邻域图像（COLMAP FindLocalBundle）
     */
    void findLocalBundle(int imageId, std::vector<int> &outImageIds) const;

    /**
     * @brief 收集参与BA的观测
     *
     * @param imageSet 相机白名单（为空表示不限）
     * @param pointSet 三维点白名单（为空表示不限）
     */
    void collectObservations(const std::set<int> &imageSet,
                             const std::set<int> &pointSet,
                             std::vector<std::tuple<int, int, cv::Point2d>> &out) const;

    /// 组装BA用的相机参数数组 [rx,ry,rz,tx,ty,tz]
    void buildCameraParams(std::vector<std::array<double, 6>> &out) const;

    /// 组装BA用的三维点数组（下标 = ObservationManager 的点 id，含墓碑位）
    void buildPointParams(std::vector<std::array<double, 3>> &out) const;

    /// 把BA优化后的相机参数写回位姿数组
    void applyCameraParams(const std::vector<std::array<double, 6>> &cameras,
                           const std::vector<char> &cameraVariable);

    /// 把BA优化后的点坐标写回 ObservationManager
    void applyPointParams(const std::vector<std::array<double, 3>> &points);

    /// 全局BA + 自标定（固定种子相机；未注册相机不参与）
    void runGlobalBundleAdjustment(bool fixIntrinsics);

    /// 补全 + 合并全部轨迹
    size_t completeAndMergeTracks();

    /// 推送当前重建状态（点云 + 已注册相机）
    void emitCurrentState(bool reset);

    /// 各子步骤累计耗时（毫秒）—— 用于定位增量重建的性能瓶颈
    struct StepTiming
    {
        double findNext = 0.0;      ///< FindNextImages（可见点数排名）
        double triangulate = 0.0;   ///< TriangulateImage
        double neighbor = 0.0;      ///< FindLocalBundle
        double localBa = 0.0;       ///< Ceres 局部 BA
        double merge = 0.0;         ///< MergeTracks
        double complete = 0.0;      ///< CompleteTracks
        double completeImg = 0.0;   ///< CompleteImage
        double filter = 0.0;        ///< FilterPoints3D
        double emitMs = 0.0;        ///< 点云推送（不能叫 emit：Qt 宏）
        void reset() { *this = StepTiming(); }
    };
    StepTiming m_timing;

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

    /// 增量三角化器参数
    TriangulatorOptions m_triOptions;

    // ── 阶段二运行时状态（仅在 triangulate() 执行期间有效）──────────────
    const std::vector<MatchPair> *m_matchPairs = nullptr;
    const std::vector<FeatureResult> *m_features = nullptr;

    std::unique_ptr<CorrespondenceGraph> m_graph;
    std::unique_ptr<ObservationManager> m_obs;
    std::unique_ptr<IncrementalTriangulator> m_triangulator;

    std::vector<CameraPose> *m_poses = nullptr;    ///< 指向输出位姿数组
    cv::Mat m_K;                                  ///< 当前内参（自标定后会更新）
    /// 当前畸变系数（5×1，k1,k2,p1,p2,k3；阶段五 5.2 由全局BA标定，全零=针孔）
    cv::Mat m_dist;
    std::set<int> m_registeredCameras;
    std::vector<int> m_numRegTrials;              ///< 每张图像的注册尝试次数
    int m_refCameraId = 0;                        ///< 世界坐标系参考相机（种子图1）

    size_t m_lastLocalChanged = 0;    ///< 上次局部BA后被合并/补全/过滤的观测数
    size_t m_lastLocalResiduals = 0;  ///< 上次局部BA参与优化的观测数
};

#endif // TRIANGULATE_H