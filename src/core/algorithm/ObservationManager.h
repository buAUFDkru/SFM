#ifndef OBSERVATIONMANAGER_H
#define OBSERVATIONMANAGER_H

#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "data/SfmTypes.h"
#include "CorrespondenceGraph.h"

/**
 * @brief 三维点与特征轨迹管理器（对齐 COLMAP sfm/observation_manager.h）
 *
 * 稀疏重建的「重建状态」全部集中在这里：
 * - 三维点存储（位置 / 颜色 / 轨迹）
 * - 关键点 → 三维点映射（kpTo3D）
 * - 轨迹增删合并（AddPoint3D / AddObservation / MergePoints3D / DeletePoint3D）
 * - 可见性统计（NumVisiblePoints3D / NumObservations）
 * - 外点过滤（重投影误差 + 三角化角）
 *
 * ### 与阶段一的本质区别
 *
 * 阶段一每轮只把「本轮新算的点」塞进 outPoints3D，从不复用、从不合并，
 * 同一个空间点会被不同图像对反复三角化。现在所有点都活在这里，
 * 新图像注册时可以复用**任意**历史已三角化点，也能被 Complete / Merge 补全合并。
 *
 * ### 存储约定
 *
 * m_points 只追加不删除，删除的点用 m_alive[i] = 0 标记（墓碑）。
 * 原因是 BA 的残差块直接引用点坐标数组的内存地址，若在优化过程中搬移元素
 * 会让 Ceres 拿到悬空指针。最终输出时由 compact() 压缩为连续数组。
 */
class ObservationManager
{
public:
    /**
     * @param features 图像特征点（提供关键点像素坐标与数量，生命周期须长于本对象）
     * @param graph 特征轨迹对应图（生命周期须长于本对象）
     */
    ObservationManager(const std::vector<FeatureResult> &features,
                       const CorrespondenceGraph &graph);

    // ──────────────────────────────────────────────────────────────────────
    //  三维点增删改
    // ──────────────────────────────────────────────────────────────────────

    /**
     * @brief 新增三维点并建立轨迹
     * @param xyz 世界坐标
     * @param track 轨迹（(imageId, keypointIdx) 列表）
     * @param color BGR 颜色
     * @return 新点 id
     */
    int addPoint3D(const cv::Point3d &xyz,
                   const std::vector<CorrespondenceGraph::Observation> &track,
                   const cv::Vec3b &color);

    /// 给已有三维点追加一条观测（同时更新 kpTo3D）
    void addObservation(int pointId, int imageId, int kptIdx);

    /// 删除某条观测；若剩余观测 < 2 则删除整个点
    void deleteObservation(int imageId, int kptIdx);

    /// 删除三维点（墓碑标记，释放 kpTo3D 中属于该点的条目）
    void deletePoint3D(int pointId);

    /**
     * @brief 合并两个三维点（COLMAP MergePoints3D）
     *
     * 位置按轨迹长度加权平均，轨迹合并到 id1 上，id2 被删除。
     * 调用方需先自行校验合并后所有观测的重投影误差（见 IncrementalTriangulator::merge）。
     * @return 合并后的点 id（= id1）
     */
    int mergePoints3D(int id1, int id2);

    bool exists(int pointId) const
    {
        return pointId >= 0 &&
               pointId < static_cast<int>(m_points.size()) &&
               m_alive[static_cast<size_t>(pointId)] != 0;
    }

    Point3D &point(int pointId) { return m_points[static_cast<size_t>(pointId)]; }
    const Point3D &point(int pointId) const { return m_points[static_cast<size_t>(pointId)]; }

    const std::vector<Point3D> &allPoints() const { return m_points; }
    const std::vector<char> &aliveMask() const { return m_alive; }
    size_t numAlivePoints() const { return m_numAlive; }

    /// 关键点是否已三角化；未三角化返回 -1
    int pointIdAt(int imageId, int kptIdx) const;

    // ──────────────────────────────────────────────────────────────────────
    //  可见性统计
    // ──────────────────────────────────────────────────────────────────────

    /// 该图像已三角化的观测数量
    size_t numObservations(int imageId) const;

    /**
     * @brief 该图像「能看到」多少个已三角化点（COLMAP NumVisiblePoints3D）
     *
     * 语义：统计该图像中有多少个关键点，其所在轨迹上**存在**已三角化的观测。
     * 由于只有已注册图像才可能产生三维点，这等价于「通过与已注册图像的匹配
     * 能看到的三维点数量」。
     *
     * 该值同时决定：
     * - 该图像能否被注册（需 ≥ abs_pose_min_num_inliers）
     * - 该图像在候选队列中的优先级（数量越多几何约束越强）
     */
    size_t numVisiblePoints3D(int imageId) const;

    /**
     * @brief 收集该图像可用的 2D-3D 对应（COLMAP RegisterNextImage 的搜索步骤）
     *
     * 遍历该图像每个关键点的轨迹，找到已三角化的对应点；
     * 每个关键点最多贡献一条对应（避免同一像素重复约束）。
     * @param outKptIdx 输出：该图像上的关键点索引
     * @param outPointId 输出：对应的三维点 id
     * @return 对应条数
     */
    size_t collect2D3DCorrespondences(int imageId,
                                      std::vector<int> &outKptIdx,
                                      std::vector<int> &outPointId) const;

    /// 统计图像对 (img1, img2) 中「至少一端已三角化」的匹配数
    size_t numTriangulatedMatches(const MatchPair &mp) const;

    /// 该图像上已三角化的三维点 id 列表（用于局部BA挑选邻域）
    void collectObservedPoints(int imageId, std::vector<int> &outPointIds) const;

    /// 全部观测条数（用于判断精化是否收敛）
    size_t totalObservations() const;

    // ──────────────────────────────────────────────────────────────────────
    //  外点过滤
    // ──────────────────────────────────────────────────────────────────────

    /**
     * @brief 过滤指定三维点（COLMAP FilterPoints3D）
     *
     * 两级判据：
     * 1. 重投影误差：单条观测误差 > maxReprojErrorPx 或深度 ≤ 0 → 删除该条观测；
     *    若待删观测数 ≥ 轨迹长度 - 1 → 删除整个点
     * 2. 三角化角：所有观测两两夹角均 < minTriAngleDeg → 删除整个点
     * @return 被剔除的观测条数
     */
    size_t filterPoints3D(double maxReprojErrorPx, double minTriAngleDeg,
                          const std::vector<int> &pointIds,
                          const std::vector<CameraPose> &poses,
                          const cv::Mat &K);

    /// 过滤全部三维点
    size_t filterAllPoints3D(double maxReprojErrorPx, double minTriAngleDeg,
                             const std::vector<CameraPose> &poses,
                             const cv::Mat &K);

    /// 只过滤「在指定图像上有观测」的三维点
    size_t filterPoints3DInImages(double maxReprojErrorPx, double minTriAngleDeg,
                                  const std::vector<int> &imageIds,
                                  const std::vector<CameraPose> &poses,
                                  const cv::Mat &K);

    // ──────────────────────────────────────────────────────────────────────
    //  输出
    // ──────────────────────────────────────────────────────────────────────

    /**
     * @brief 压缩输出：去掉墓碑点，重新编号
     * @param outPoints 输出：连续的三维点数组（id 已重排为数组下标）
     * @param outIdMap 可选输出：旧 id → 新 id（-1 表示已删除）
     */
    void compact(std::vector<Point3D> &outPoints,
                 std::vector<int> *outIdMap = nullptr) const;

    /// 该图像的关键点数量（来自 features）
    int numKeypoints(int imageId) const;

    /// 关键点像素坐标
    cv::Point2d keypointPos(int imageId, int kptIdx) const;

private:
    const std::vector<FeatureResult> *m_features = nullptr;
    const CorrespondenceGraph *m_graph = nullptr;

    std::vector<Point3D> m_points;                       ///< 含墓碑的点存储
    std::vector<char> m_alive;                           ///< 1 = 有效
    std::vector<std::unordered_map<int, int>> m_kpTo3D;  ///< imageId → (kptIdx → pointId)
    size_t m_numAlive = 0;
};

#endif // OBSERVATIONMANAGER_H
