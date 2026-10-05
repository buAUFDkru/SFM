/**
 * @file CorrespondenceGraph.cpp
 * @brief 特征轨迹对应图实现 —— 用并查集把匹配结果合并为特征轨迹
 *
 * 对齐 COLMAP scene/correspondence_graph.h：
 * - AddCorrespondences → build()：逐条匹配做 union
 * - FindCorrespondences → findCorrespondences()
 * - ExtractTransitiveCorrespondences → extractTransitiveCorrespondences()
 * - IsTwoViewObservation → isTwoViewObservation()
 */

#include "CorrespondenceGraph.h"

#include <algorithm>
#include <climits>
#include <numeric>

#include <QDebug>
#include <QString>

void CorrespondenceGraph::build(const std::vector<MatchPair> &matchPairs,
                                const std::vector<int> &numKeypointsPerImage)
{
    m_numImages = static_cast<int>(numKeypointsPerImage.size());
    m_offset.assign(static_cast<size_t>(m_numImages) + 1, 0);
    for (int i = 0; i < m_numImages; ++i) {
        m_offset[static_cast<size_t>(i) + 1] =
            m_offset[static_cast<size_t>(i)] + std::max(0, numKeypointsPerImage[static_cast<size_t>(i)]);
    }

    const int numNodes = m_offset[static_cast<size_t>(m_numImages)];
    m_parent.resize(static_cast<size_t>(numNodes));
    std::iota(m_parent.begin(), m_parent.end(), 0);

    // ── 逐条匹配做 union ──────────────────────────────────────────────────
    for (const auto &mp : matchPairs) {
        if (mp.imageId1 < 0 || mp.imageId1 >= m_numImages) continue;
        if (mp.imageId2 < 0 || mp.imageId2 >= m_numImages) continue;

        const int maxKp1 = numKeypointsPerImage[static_cast<size_t>(mp.imageId1)];
        const int maxKp2 = numKeypointsPerImage[static_cast<size_t>(mp.imageId2)];

        for (const auto &m : mp.matches) {
            const int q = m.queryIdx;
            const int t = m.trainIdx;
            if (q < 0 || q >= maxKp1 || t < 0 || t >= maxKp2) continue;
            unite(nodeId(mp.imageId1, q), nodeId(mp.imageId2, t));
        }
    }

    // ── 统计每个连通分量的规模 ────────────────────────────────────────────
    std::vector<int> rootSize(static_cast<size_t>(numNodes), 0);
    for (int i = 0; i < numNodes; ++i) {
        rootSize[static_cast<size_t>(findRoot(i))] += 1;
    }

    // ── 只保留规模 ≥ 2 的轨迹 ─────────────────────────────────────────────
    m_compId.assign(static_cast<size_t>(numNodes), -1);
    m_tracks.clear();
    std::vector<int> rootToComp(static_cast<size_t>(numNodes), -1);

    for (int i = 0; i < numNodes; ++i) {
        const int r = findRoot(i);
        if (rootSize[static_cast<size_t>(r)] < 2) continue;

        int &comp = rootToComp[static_cast<size_t>(r)];
        if (comp < 0) {
            comp = static_cast<int>(m_tracks.size());
            m_tracks.emplace_back();
            m_tracks.back().reserve(static_cast<size_t>(rootSize[static_cast<size_t>(r)]));
        }
        m_compId[static_cast<size_t>(i)] = comp;
    }

    // ── 把节点解码回 (imageId, keypointIdx) ───────────────────────────────
    for (int img = 0; img < m_numImages; ++img) {
        const int begin = m_offset[static_cast<size_t>(img)];
        const int end   = m_offset[static_cast<size_t>(img) + 1];
        for (int n = begin; n < end; ++n) {
            const int comp = m_compId[static_cast<size_t>(n)];
            if (comp < 0) continue;
            m_tracks[static_cast<size_t>(comp)].emplace_back(img, n - begin);
        }
    }

    // ── 规模诊断：并查集在高重叠数据上会形成巨型连通块 ────────────────────
    size_t maxSize = 0;
    size_t sumSize = 0;
    size_t multiImage = 0;   // 含「同一张图多个观测」的轨迹数
    for (const auto &track : m_tracks) {
        maxSize = std::max(maxSize, track.size());
        sumSize += track.size();
        if (track.size() > static_cast<size_t>(m_numImages)) ++multiImage;
    }
    m_imgSeen.assign(static_cast<size_t>(m_numImages), 0);
    m_visitStamp.assign(static_cast<size_t>(numNodes), 0);
    m_visitEpoch = 0;

    qDebug().noquote()
        << QString("[对应图] 节点=%1, 轨迹=%2, 最大轨迹=%3, 平均=%4, 超长轨迹(>%5)=%6")
               .arg(numNodes)
               .arg(static_cast<qlonglong>(m_tracks.size()))
               .arg(static_cast<qlonglong>(maxSize))
               .arg(m_tracks.empty() ? 0.0
                                     : static_cast<double>(sumSize) /
                                           static_cast<double>(m_tracks.size()), 0, 'f', 2)
               .arg(m_numImages)
               .arg(static_cast<qlonglong>(multiImage));
}

bool CorrespondenceGraph::hasCorrespondences(int imageId, int kptIdx) const
{
    if (imageId < 0 || imageId >= m_numImages) return false;
    if (kptIdx  < 0 || kptIdx  >= m_offset[static_cast<size_t>(imageId) + 1] -
                                   m_offset[static_cast<size_t>(imageId)]) {
        return false;
    }
    return m_compId[static_cast<size_t>(nodeId(imageId, kptIdx))] >= 0;
}

void CorrespondenceGraph::findCorrespondences(int imageId, int kptIdx,
                                              std::vector<Observation> *out) const
{
    out->clear();
    if (!hasCorrespondences(imageId, kptIdx)) return;

    const auto &track = m_tracks[static_cast<size_t>(m_compId[static_cast<size_t>(nodeId(imageId, kptIdx))])];

    // 一张图至多贡献一个观测（详见头文件注释）：候选集上界 = 图像数，
    // 巨型连通块下也不会把三角化/补全拖成 O(T^2)~O(T^3)。
    if (m_imgSeen.size() != static_cast<size_t>(m_numImages)) {
        m_imgSeen.assign(static_cast<size_t>(m_numImages), 0);
    }
    std::fill(m_imgSeen.begin(), m_imgSeen.end(), 0);

    out->reserve(std::min<size_t>(track.size(), static_cast<size_t>(m_numImages)));
    for (const auto &obs : track) {
        if (obs.first == imageId) continue;
        if (obs.first < 0 || obs.first >= m_numImages) continue;
        if (m_imgSeen[static_cast<size_t>(obs.first)]) continue;
        m_imgSeen[static_cast<size_t>(obs.first)] = 1;
        out->push_back(obs);
    }
}

void CorrespondenceGraph::extractTransitiveCorrespondences(
    int imageId, int kptIdx, int transitivity, std::vector<Observation> *out) const
{
    if (transitivity <= 1) {
        findCorrespondences(imageId, kptIdx, out);
        return;
    }

    out->clear();
    if (!hasCorrespondences(imageId, kptIdx)) return;

    // 与 COLMAP 一致：BFS 按「层」扩展，每层只扩展上一层新加入的观测。
    // 访问标记用 epoch 递增的整型数组（O(1)），旧实现用 vector 线性扫描，
    // 单次查询是 O(V^2)，大轨迹上同样会退化。
    if (m_visitStamp.size() != static_cast<size_t>(m_offset[static_cast<size_t>(m_numImages)])) {
        m_visitStamp.assign(static_cast<size_t>(m_offset[static_cast<size_t>(m_numImages)]), 0);
        m_visitEpoch = 0;
    }
    if (m_visitEpoch >= INT_MAX) {
        std::fill(m_visitStamp.begin(), m_visitStamp.end(), 0);
        m_visitEpoch = 0;
    }
    ++m_visitEpoch;
    const int epoch = m_visitEpoch;

    std::vector<Observation> level;
    level.emplace_back(imageId, kptIdx);
    m_visitStamp[static_cast<size_t>(nodeId(imageId, kptIdx))] = epoch;

    std::vector<Observation> scratch;

    for (int t = 0; t < transitivity; ++t) {
        std::vector<Observation> nextLevel;

        for (const auto &ref : level) {
            scratch.clear();
            findCorrespondences(ref.first, ref.second, &scratch);
            for (const auto &c : scratch) {
                int &stamp = m_visitStamp[static_cast<size_t>(nodeId(c.first, c.second))];
                if (stamp == epoch) continue;
                stamp = epoch;
                out->push_back(c);
                nextLevel.push_back(c);
            }
        }

        if (nextLevel.empty()) break;
        level = std::move(nextLevel);
    }
}

bool CorrespondenceGraph::isTwoViewObservation(int imageId, int kptIdx) const
{
    if (!hasCorrespondences(imageId, kptIdx)) return true;

    const auto &track = m_tracks[static_cast<size_t>(m_compId[static_cast<size_t>(nodeId(imageId, kptIdx))])];
    if (track.size() < 2) return true;

    int firstImage = track.front().first;
    for (const auto &obs : track) {
        if (obs.first != firstImage) {
            // 出现第二张图像；检查是否还有第三张
            for (const auto &o2 : track) {
                if (o2.first != firstImage && o2.first != obs.first) return false;
            }
            return true;
        }
    }
    return true;
}
