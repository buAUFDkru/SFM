#ifndef CORRESPONDENCEGRAPH_H
#define CORRESPONDENCEGRAPH_H

#include <utility>
#include <vector>

#include "data/SfmTypes.h"

/**
 * @brief 特征轨迹对应图（对齐 COLMAP scene/correspondence_graph.h）
 *
 * 把所有图像对的匹配结果合并为「特征轨迹（track）」：
 * 一条轨迹是一组互为匹配的关键点 (imageId, keypointIdx)，理论上对应同一个空间点。
 *
 * ### 为什么必须要有这一层
 *
 * 阶段一的 Triangulate 只能在同一轮里用「当前两张已注册图的共有点」现算新点，
 * 从不跨轮复用，导致：
 * - 第 N 张图注册时无法利用前 N-1 张图已经三角化好的点（只能靠本轮新点凑数）
 * - 同一个空间点会被不同图像对重复三角化，产生大量重复点
 *
 * 有了对应图之后，才能按 COLMAP 的方式实现：
 * - NumVisiblePoints3D：某张未注册图「能看到」多少个已三角化点
 * - TriangulateImage：Find → Create / Continue（新建点 + 续接已有轨迹）
 * - Complete / Merge / Retriangulate：补全与合并轨迹
 *
 * ### 实现
 *
 * 用并查集（union-find）把关键点合并为连通分量，每个连通分量即一条轨迹。
 * 只保存规模 ≥ 2 的轨迹，孤立关键点不占内存。
 *
 * 节点编号：node = offset[imageId] + keypointIdx
 */
class CorrespondenceGraph
{
public:
    /// 观测：某张图像上的某个关键点
    using Observation = std::pair<int, int>;  // (imageId, keypointIdx)

    /**
     * @brief 用匹配对构建轨迹图
     *
     * @param matchPairs 所有图像对的匹配结果
     * @param numKeypointsPerImage 每张图像的关键点数量（下标 = imageId）
     */
    void build(const std::vector<MatchPair> &matchPairs,
               const std::vector<int> &numKeypointsPerImage);

    /// 图像数量
    int numImages() const { return m_numImages; }

    /// 轨迹数量（规模 ≥ 2）
    int numTracks() const { return static_cast<int>(m_tracks.size()); }

    /// 该关键点是否参与了任何匹配
    bool hasCorrespondences(int imageId, int kptIdx) const;

    /**
     * @brief 直接对应（transitivity = 1）
     *
     * 同一条轨迹中来自**其它图像**的所有观测（不含查询观测自身）。
     */
    void findCorrespondences(int imageId, int kptIdx,
                             std::vector<Observation> *out) const;

    /**
     * @brief 传递对应（COLMAP ExtractTransitiveCorrespondences）
     *
     * 以 (imageId, kptIdx) 为起点做 BFS：第 t 层 = 与第 t-1 层任一观测直接对应的
     * 其它图像观测。transitivity = 1 等价于 findCorrespondences。
     *
     * 注意：默认 max_transitivity = 1 —— 只使用直接对应，避免传递性带来的错误匹配。
     *
     * @param transitivity 最大层数（≥1）
     */
    void extractTransitiveCorrespondences(int imageId, int kptIdx,
                                          int transitivity,
                                          std::vector<Observation> *out) const;

    /**
     * @brief 该观测所在的轨迹是否只横跨两张图像
     *
     * COLMAP IsTwoViewObservation。两视图轨迹缺乏冗余约束，误匹配无法被
     * 三角化角/重投影误差剔除，因此默认 ignore_two_view_tracks = true 时不会被
     * 用来新建三维点（但可以被 Complete 续接到已有三维点上）。
     */
    bool isTwoViewObservation(int imageId, int kptIdx) const;

private:
    int m_numImages = 0;
    std::vector<int> m_offset;                     ///< 节点偏移：node = offset[img] + kpt
    mutable std::vector<int> m_parent;             ///< 并查集父节点（const 查询时做路径压缩）
    std::vector<int> m_compId;                     ///< node → 轨迹号；< 0 表示孤立点
    std::vector<std::vector<Observation>> m_tracks;///< 轨迹号 → 观测列表

    int nodeId(int imageId, int kptIdx) const
    {
        return m_offset[imageId] + kptIdx;
    }

    int findRoot(int x) const
    {
        while (m_parent[x] != x) {
            m_parent[x] = m_parent[m_parent[x]];  // 路径压缩
            x = m_parent[x];
        }
        return x;
    }

    void unite(int a, int b)
    {
        const int ra = findRoot(a);
        const int rb = findRoot(b);
        if (ra != rb) m_parent[ra] = rb;
    }
};

#endif // CORRESPONDENCEGRAPH_H
