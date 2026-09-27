#include "BuiltinMatchers.h"

#include <QDebug>

#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

// ── FLANN ───────────────────────────────────────────────────────────────────

namespace {

cv::Ptr<cv::FlannBasedMatcher> makeFlannMatcher(int trees, int descType)
{
    if (descType == CV_8U) {
        // 二进制描述符（ORB 等）→ LSH 索引
        return cv::makePtr<cv::FlannBasedMatcher>(
            cv::makePtr<cv::flann::LshIndexParams>(20, 10, 2));
    }
    // 浮点描述符（SIFT/SURF 等）→ KD 树索引
    return cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(trees));
}

} // namespace

FlannMatcherWorker::FlannMatcherWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("trees")))
        m_trees = params.value(QStringLiteral("trees")).toInt(4);
    if (params.contains(QStringLiteral("checks")))
        m_checks = params.value(QStringLiteral("checks")).toInt(50);
}

bool FlannMatcherWorker::knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                      std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                                      int k)
{
    outKnnMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto flannMatcher = makeFlannMatcher(m_trees, desc1.type());
    flannMatcher->knnMatch(desc1, desc2, outKnnMatches, k);
    return !outKnnMatches.empty();
}

bool FlannMatcherWorker::matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                   std::vector<cv::DMatch> &outMatches)
{
    outMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto flannMatcher = makeFlannMatcher(m_trees, desc1.type());
    flannMatcher->match(desc1, desc2, outMatches);
    return !outMatches.empty();
}

QList<ParamSchema> FlannMatcherPlugin::paramSchema() const
{
    ParamSchema trees;
    trees.name = QStringLiteral("trees");
    trees.type = ParamSchema::Type::Int;
    trees.defaultValue = 4;
    trees.min = 1;
    trees.max = 32;
    trees.group = QStringLiteral("FLANN");
    trees.tooltip = QStringLiteral("KD 树的棵数");

    ParamSchema checks;
    checks.name = QStringLiteral("checks");
    checks.type = ParamSchema::Type::Int;
    checks.defaultValue = 50;
    checks.min = 1;
    checks.max = 200;
    checks.group = QStringLiteral("FLANN");
    checks.tooltip = QStringLiteral("搜索时回溯节点次数(预留，当前实现未使用)");

    return { trees, checks };
}

bool FlannMatcherPlugin::validateParams(const QJsonObject &params,
                                        QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("trees") ||
            key == QStringLiteral("checks")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else {
            if (errorMsg)
                *errorMsg = QStringLiteral("未知参数 %1").arg(key);
            return false;
        }
    }
    return true;
}

std::unique_ptr<IMatcherWorker> FlannMatcherPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[FLANN] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<FlannMatcherWorker>(params);
}

// ── BruteForce ──────────────────────────────────────────────────────────────

BruteForceMatcherWorker::BruteForceMatcherWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("normType")))
        m_normType = params.value(QStringLiteral("normType")).toInt(-1);
    if (params.contains(QStringLiteral("crossCheck")))
        m_crossCheck = params.value(QStringLiteral("crossCheck")).toBool(false);
}

int BruteForceMatcherWorker::autoNormType(const cv::Mat &desc) const
{
    if (m_normType >= 0)
        return m_normType;
    // 根据描述符数据类型自动选择距离度量
    // CV_8U  → 二进制描述符(ORB 等) → Hamming 距离
    // 其他   → 浮点描述符(SIFT/SURF 等) → L2 距离
    return (desc.type() == CV_8U) ? cv::NORM_HAMMING : cv::NORM_L2;
}

bool BruteForceMatcherWorker::knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                           std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                                           int k)
{
    outKnnMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto bfMatcher = cv::BFMatcher::create(autoNormType(desc1), m_crossCheck);
    bfMatcher->knnMatch(desc1, desc2, outKnnMatches, k);
    return !outKnnMatches.empty();
}

bool BruteForceMatcherWorker::matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                        std::vector<cv::DMatch> &outMatches)
{
    outMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    auto bfMatcher = cv::BFMatcher::create(autoNormType(desc1), m_crossCheck);
    bfMatcher->match(desc1, desc2, outMatches);
    return !outMatches.empty();
}

QList<ParamSchema> BruteForceMatcherPlugin::paramSchema() const
{
    ParamSchema normType;
    normType.name = QStringLiteral("normType");
    normType.type = ParamSchema::Type::Int;
    normType.defaultValue = static_cast<int>(cv::NORM_L2);
    normType.min = -1;
    normType.max = 7;
    normType.group = QStringLiteral("BruteForce");
    // OpenCV NormTypes: NORM_INF=1, NORM_L1=2, NORM_L2=4, NORM_L2SQR=5, NORM_HAMMING=6, NORM_HAMMING2=7
    normType.tooltip = QStringLiteral("距离度量：4=NORM_L2(默认,浮点描述子)；1=NORM_INF 2=NORM_L1 5=NORM_L2SQR 6=NORM_HAMMING 7=NORM_HAMMING2(二进制描述子)；-1=按描述符类型自动选择(二进制→Hamming，浮点→L2)");

    ParamSchema crossCheck;
    crossCheck.name = QStringLiteral("crossCheck");
    crossCheck.type = ParamSchema::Type::Bool;
    crossCheck.defaultValue = false;
    crossCheck.group = QStringLiteral("BruteForce");
    crossCheck.tooltip = QStringLiteral("是否启用交叉验证，仅返回双向一致的匹配");

    return { normType, crossCheck };
}

bool BruteForceMatcherPlugin::validateParams(const QJsonObject &params,
                                             QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("normType")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("crossCheck")) {
            if (!params.value(key).isBool()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 bool").arg(key);
                return false;
            }
        } else {
            if (errorMsg)
                *errorMsg = QStringLiteral("未知参数 %1").arg(key);
            return false;
        }
    }
    return true;
}

std::unique_ptr<IMatcherWorker> BruteForceMatcherPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[BruteForce] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<BruteForceMatcherWorker>(params);
}

// ── NearestNeighbor ─────────────────────────────────────────────────────────

namespace {

/// 行 L2 归一化（幂等：已归一化的输入不变）；零范数行保持零（相似度按 0 处理）
void nnNormalizeRows(const cv::Mat &desc, cv::Mat &out)
{
    cv::Mat f;
    desc.convertTo(f, CV_32F);
    out = f.clone();
    for (int i = 0; i < out.rows; ++i) {
        float *row = out.ptr<float>(i);
        double sq = 0.0;
        for (int j = 0; j < out.cols; ++j)
            sq += static_cast<double>(row[j]) * row[j];
        const float norm = static_cast<float>(std::sqrt(sq));
        if (norm > 0.0f) {
            for (int j = 0; j < out.cols; ++j)
                row[j] /= norm;
        }
    }
}

/// 余弦相似度矩阵 sim = d1n * d2n^T（m×n CV_32F）
void nnCosineSimilarity(const cv::Mat &d1n, const cv::Mat &d2n, cv::Mat &sim)
{
    sim = d1n * d2n.t();
}

/// 余弦相似度 → 欧氏距离（L2 归一化向量: ||a-b||^2 = 2(1-cos)），与 FLANN/BF 距离语义一致
inline float nnSimToDistance(float sim)
{
    return std::sqrt(std::max(0.0f, 2.0f * (1.0f - sim)));
}

} // namespace

NearestNeighborMatcherWorker::NearestNeighborMatcherWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("mutualCheck")))
        m_mutualCheck = params.value(QStringLiteral("mutualCheck")).toBool(true);
    if (params.contains(QStringLiteral("distanceThreshold")))
        m_distanceThreshold = params.value(QStringLiteral("distanceThreshold")).toDouble(0.0);
}

bool NearestNeighborMatcherWorker::knnMatchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                                std::vector<std::vector<cv::DMatch>> &outKnnMatches,
                                                int k)
{
    outKnnMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    cv::Mat d1n, d2n, sim;
    nnNormalizeRows(desc1, d1n);
    nnNormalizeRows(desc2, d2n);
    nnCosineSimilarity(d1n, d2n, sim);

    const int m = sim.rows, n = sim.cols;
    const int kk = std::min(std::max(k, 1), n);

    // 列方向 top-1（mutual check 需要反向最佳）
    std::vector<int> bestRowForCol;
    if (m_mutualCheck) {
        bestRowForCol.assign(static_cast<size_t>(n), -1);
        for (int j = 0; j < n; ++j) {
            int best = -1;
            float bestVal = -2.0f;
            for (int i = 0; i < m; ++i) {
                const float v = sim.at<float>(i, j);
                if (v > bestVal) { bestVal = v; best = i; }
            }
            bestRowForCol[static_cast<size_t>(j)] = best;
        }
    }

    outKnnMatches.reserve(static_cast<size_t>(m));
    std::vector<int> idx(kk);
    std::vector<float> val(kk);
    for (int i = 0; i < m; ++i) {
        // 行内 top-k：降序插入选择（k 通常为 1 或 2）
        std::fill(idx.begin(), idx.end(), -1);
        std::fill(val.begin(), val.end(), -2.0f);
        const float *row = sim.ptr<float>(i);
        for (int j = 0; j < n; ++j) {
            const float v = row[j];
            if (v <= val[kk - 1] && idx[kk - 1] != -1)
                continue;
            int pos = kk - 1;
            while (pos > 0 && val[pos - 1] < v) {
                val[pos] = val[pos - 1];
                idx[pos] = idx[pos - 1];
                --pos;
            }
            val[pos] = v;
            idx[pos] = j;
        }

        // mutual check（对齐 hloc 语义：行 top-1 未被反向认可 → 该查询视为无匹配）
        if (m_mutualCheck &&
            (idx[0] < 0 || bestRowForCol[static_cast<size_t>(idx[0])] != i))
            continue;

        std::vector<cv::DMatch> candidates;
        candidates.reserve(static_cast<size_t>(kk));
        for (int t = 0; t < kk; ++t) {
            if (idx[t] < 0)
                continue;
            const float dist = nnSimToDistance(val[t]);
            if (m_distanceThreshold > 0.0 && dist > m_distanceThreshold)
                continue;
            candidates.emplace_back(i, idx[t], dist);
        }
        if (!candidates.empty())
            outKnnMatches.push_back(std::move(candidates));
    }
    return !outKnnMatches.empty();
}

bool NearestNeighborMatcherWorker::matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                             std::vector<cv::DMatch> &outMatches)
{
    outMatches.clear();
    if (desc1.empty() || desc2.empty())
        return false;

    cv::Mat d1n, d2n, sim;
    nnNormalizeRows(desc1, d1n);
    nnNormalizeRows(desc2, d2n);
    nnCosineSimilarity(d1n, d2n, sim);

    const int m = sim.rows, n = sim.cols;

    std::vector<int> bestRowForCol;
    if (m_mutualCheck) {
        bestRowForCol.assign(static_cast<size_t>(n), -1);
        for (int j = 0; j < n; ++j) {
            int best = -1;
            float bestVal = -2.0f;
            for (int i = 0; i < m; ++i) {
                const float v = sim.at<float>(i, j);
                if (v > bestVal) { bestVal = v; best = i; }
            }
            bestRowForCol[static_cast<size_t>(j)] = best;
        }
    }

    for (int i = 0; i < m; ++i) {
        const float *row = sim.ptr<float>(i);
        int bestJ = -1;
        float bestVal = -2.0f;
        for (int j = 0; j < n; ++j) {
            if (row[j] > bestVal) { bestVal = row[j]; bestJ = j; }
        }
        if (bestJ < 0)
            continue;
        if (m_mutualCheck && bestRowForCol[static_cast<size_t>(bestJ)] != i)
            continue;
        const float dist = nnSimToDistance(bestVal);
        if (m_distanceThreshold > 0.0 && dist > m_distanceThreshold)
            continue;
        outMatches.emplace_back(i, bestJ, dist);
    }
    return !outMatches.empty();
}

QList<ParamSchema> NearestNeighborMatcherPlugin::paramSchema() const
{
    ParamSchema mutualCheck;
    mutualCheck.name = QStringLiteral("mutualCheck");
    mutualCheck.type = ParamSchema::Type::Bool;
    mutualCheck.defaultValue = true;
    mutualCheck.group = QStringLiteral("NearestNeighbor");
    mutualCheck.tooltip = QStringLiteral("双向一致性校验：两图互为最佳匹配才保留（对齐 hloc mutual_check）");

    ParamSchema distanceThreshold;
    distanceThreshold.name = QStringLiteral("distanceThreshold");
    distanceThreshold.type = ParamSchema::Type::Double;
    distanceThreshold.defaultValue = 0.0;
    distanceThreshold.min = 0.0;
    distanceThreshold.max = 2.0;
    distanceThreshold.group = QStringLiteral("NearestNeighbor");
    distanceThreshold.tooltip = QStringLiteral("距离阈值 sqrt(2*(1-cos))（归一化向量欧氏距离），大于该值的匹配被剔除；0 表示不过滤");

    return { mutualCheck, distanceThreshold };
}

bool NearestNeighborMatcherPlugin::validateParams(const QJsonObject &params,
                                                  QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("mutualCheck")) {
            if (!params.value(key).isBool()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 bool").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("distanceThreshold")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 double").arg(key);
                return false;
            }
        } else {
            if (errorMsg)
                *errorMsg = QStringLiteral("未知参数 %1").arg(key);
            return false;
        }
    }
    return true;
}

std::unique_ptr<IMatcherWorker> NearestNeighborMatcherPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[NearestNeighbor] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<NearestNeighborMatcherWorker>(params);
}
