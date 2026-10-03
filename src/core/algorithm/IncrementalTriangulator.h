#ifndef INCREMENTALTRIANGULATOR_H
#define INCREMENTALTRIANGULATOR_H

#include <set>
#include <vector>

#include <opencv2/core.hpp>

#include "data/SfmTypes.h"
#include "CorrespondenceGraph.h"
#include "ObservationManager.h"

/**
 * @brief 增量三角化器（对齐 COLMAP sfm/incremental_triangulator.h）
 *
 * 负责「新图像注册后如何产出/维护三维点」，四种操作：
 *
 * | 操作 | 作用 | 调用时机 |
 * |---|---|---|
 * | TriangulateImage | Find → Create（新建点）/ Continue（续接已有轨迹） | 每注册一张新图后 |
 * | CompleteImage / Complete | 给已有三维点补观测，或用宽松阈值补建点 | 局部BA后、重三角化 |
 * | Merge | 合并被新图像「桥接」起来的两条轨迹 | 局部BA后 |
 * | Retriangulate | 对欠重建图像对（三角化率 < 20%）重新尝试 | 全局BA前 |
 *
 * ### 与阶段一的关键差异
 *
 * 阶段一只能拿「本轮两张已注册图的共有点」现算新点；现在：
 * - 一条特征轨迹上的观测只要**任意一个**已三角化，其余观测都可通过 Continue 续接，
 *   不需要凑够三视图共有点
 * - 新建点用多视图 DLT + 角度残差筛选，支持 3+ 视图冗余校验
 * - 点被统一放在 ObservationManager 中，天然去重
 */
struct TriangulatorOptions
{
    // ── 三角化质量控制 ────────────────────────────────────────────────────
    double minAngleDeg = 1.5;              ///< 最小三角化角度（度）；角度过小深度不可靠
    double createMaxAngleErrorDeg = 2.0;   ///< 新建点时允许的角度残差（度）
    double continueMaxAngleErrorDeg = 2.0; ///< 续接已有轨迹时允许的角度残差（度）

    // ── 轨迹补全与合并 ────────────────────────────────────────────────────
    double mergeMaxReprojErrorPx = 4.0;    ///< 合并两条轨迹时允许的重投影误差（像素）
    double completeMaxReprojErrorPx = 4.0; ///< 补全轨迹时允许的重投影误差（像素）
    int    completeMaxTransitivity = 5;    ///< 补全时的最大传递层数

    // ── 重三角化 ──────────────────────────────────────────────────────────
    double reMaxAngleErrorDeg = 5.0;       ///< 重三角化时放宽的角度残差（度）
    double reMinRatio = 0.2;               ///< 三角化率低于此值的图像对视为欠重建
    int    reMaxTrials = 1;                ///< 每个图像对的重三角化次数上限

    // ── 搜索范围 ──────────────────────────────────────────────────────────
    int  maxTransitivity = 1;       ///< TriangulateImage 的传递层数（1 = 只用直接对应）
    bool ignoreTwoViewTracks = true;///< 是否忽略只横跨两张图像的轨迹（COLMAP 默认 true）
    int  minTrackLength = 2;        ///< 接受一个三维点所需的最少观测数
};

class IncrementalTriangulator
{
public:
    /**
     * @param graph 特征轨迹对应图
     * @param obs 三维点管理器（会被本类修改）
     * @param features 图像特征（用于取像素坐标与颜色）
     * @param poses 相机位姿（会被本类读取；注册状态决定是否参与三角化）
     * @param K 相机内参（生命周期须长于本对象；自标定后由 setK 更新）
     */
    IncrementalTriangulator(const CorrespondenceGraph &graph,
                            ObservationManager &obs,
                            const std::vector<FeatureResult> &features,
                            std::vector<CameraPose> &poses,
                            const cv::Mat &K);

    /// 自标定后更新内参（K 由外部持有，这里只换指针）
    void setK(const cv::Mat &K) { m_K = &K; }
    const cv::Mat &K() const { return *m_K; }

    /**
     * @brief 三角化某张已注册图像上的所有观测（COLMAP TriangulateImage）
     * @return 新增/续接的观测条数
     */
    size_t triangulateImage(const TriangulatorOptions &options, int imageId);

    /// 补全某张图像上的观测（COLMAP CompleteImage）
    size_t completeImage(const TriangulatorOptions &options, int imageId);

    /// 补全指定三维点的轨迹（COLMAP CompleteTracks）
    size_t completeTracks(const TriangulatorOptions &options,
                          const std::vector<int> &pointIds);

    /// 补全全部三维点的轨迹
    size_t completeAllTracks(const TriangulatorOptions &options);

    /// 合并指定三维点的轨迹（COLMAP MergeTracks）
    size_t mergeTracks(const TriangulatorOptions &options,
                       const std::vector<int> &pointIds);

    /// 合并全部三维点的轨迹
    size_t mergeAllTracks(const TriangulatorOptions &options);

    /**
     * @brief 对欠重建图像对重三角化（COLMAP Retriangulate）
     *
     * 欠重建判据：图像对的三角化率 = 已三角化匹配数 / 总匹配数 < reMinRatio。
     * 这种情况通常出现在重建漂移之后 —— 位姿修正后原本失败的对应可以补上。
     */
    size_t retriangulate(const TriangulatorOptions &options,
                         const std::vector<MatchPair> &matchPairs);

    /// 本轮被新建/修改的点（供局部BA挑选可变点、供过滤使用）
    const std::set<int> &modifiedPoints() const { return m_modified; }
    void clearModified() { m_modified.clear(); }
    void addModifiedPoint(int pointId) { m_modified.insert(pointId); }

private:
    /// 一条候选观测（轨迹元素 + 是否已三角化 + 像素坐标）
    struct CorrData
    {
        int imageId = -1;
        int kptIdx = -1;
        int pointId = -1;      ///< -1 表示尚未三角化
        cv::Point2d xy;        ///< 像素坐标
    };

    /// Find：收集 (imageId, kptIdx) 在其它已注册图像上的对应
    size_t find(int imageId, int kptIdx, int transitivity,
                std::vector<CorrData> *out) const;

    /// Create：用未三角化的观测新建一个三维点
    size_t create(const TriangulatorOptions &options, const std::vector<CorrData> &corrs);

    /// Continue：把参考观测续接到最优的已有三维点上
    size_t continueTrack(const TriangulatorOptions &options,
                         const CorrData &ref,
                         const std::vector<CorrData> &corrs);

    /// Merge：合并与 pointId 冲突的另一条轨迹
    size_t merge(const TriangulatorOptions &options, int pointId);

    /// Complete：给 pointId 的轨迹补观测（BFS，最多 completeMaxTransitivity 层）
    size_t complete(const TriangulatorOptions &options, int pointId);

    /**
     * @brief 多视图三角化估计
     *
     * 1. 用全部观测做 DLT 求解
     * 2. 按角度残差（或像素重投影误差）+ 深度 > 0 筛选内点
     * 3. 用内点重新求解，迭代两轮
     * 4. 校验最大三角化角 ≥ minAngleDeg
     */
    bool estimateTriangulation(const std::vector<CorrData> &corrs,
                               double minAngleDeg,
                               double maxError,
                               bool errorIsPixel,
                               cv::Point3d &outXyz,
                               std::vector<char> &outInlierMask) const;

    /// 从图像上采样关键点处的 BGR 颜色
    cv::Vec3b sampleColor(int imageId, int kptIdx) const;

    const CorrespondenceGraph *m_graph = nullptr;
    ObservationManager *m_obs = nullptr;
    const std::vector<FeatureResult> *m_features = nullptr;
    std::vector<CameraPose> *m_poses = nullptr;
    const cv::Mat *m_K = nullptr;

    std::set<int> m_modified;                    ///< 本轮新建/修改的点
    std::set<std::pair<int, int>> m_mergeTrials; ///< 已尝试过的轨迹合并组合（防重复）
    std::map<std::pair<int, int>, int> m_reTrials; ///< 图像对 → 重三角化次数
};

#endif // INCREMENTALTRIANGULATOR_H
