#include "BuiltinMatchers.h"

#include <QDebug>
#include <future>

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

#include "algorithm/ThreadPool.h"

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

/// 余弦相似度 → 欧氏距离（L2 归一化向量: ||a-b||^2 = 2(1-cos)），与 FLANN/BF 距离语义一致
inline float nnSimToDistance(float sim)
{
    return std::sqrt(std::max(0.0f, 2.0f * (1.0f - sim)));
}

// ── 分块加速实现 ────────────────────────────────────────────────────────────
// 旧实现一次性计算 N1×N2 相似度矩阵（8192² 特征时 256MB）+ 单线程 cv::gemm
// + 逐列跨行跳步扫描 mutual，三处瓶颈。分块后：
//   - 相似度只保留 512×512 块缓冲（1MB/线程，L2 友好）
//   - SIMD 点积 + 行块经全局线程池并行
//   - mutual 列向 top-1 按行块局部计算后合并（严格大于 → 行号最小优先，与全量扫描一致）

constexpr int kNnTileRows = 512;
constexpr int kNnTileCols = 512;

/// SIMD 友好的 D 维点积（cv universal intrinsics，编译期自动适配 SSE/AVX 宽度）
inline float nnDotProduct(const float *a, const float *b, int d)
{
    cv::v_float32 sum = cv::vx_setzero_f32();
    int k = 0;
    for (; k + static_cast<int>(cv::v_float32::nlanes) <= d;
         k += static_cast<int>(cv::v_float32::nlanes))
        sum = cv::v_muladd(cv::v_load(a + k), cv::v_load(b + k), sum);
    float out = cv::v_reduce_sum(sum);
    for (; k < d; ++k)
        out += a[k] * b[k];
    return out;
}

/// 单个行块任务的累积结果
struct NnTileResult
{
    std::vector<int> topIdx;      ///< rows×kk，每行累积 top-k 列号（-1 为空位）
    std::vector<float> topVal;    ///< rows×kk，对应的相似度值
    std::vector<int> colBestIdx;  ///< n，列向 top-1 的行号（mutualCheck=false 时空）
    std::vector<float> colBestVal;///< n，列向 top-1 的相似度值
};

/// 处理行块 [bi0, bi1)：跨所有列块流式计算 top-k 归并与局部列向 top-1。
/// 插入选择的元素访问顺序（全局列号 j 递增）与全量扫描一致，结果等价。
NnTileResult nnMatchRowsTile(const cv::Mat &d1n, const cv::Mat &d2n,
                             int bi0, int bi1, int kk, bool mutualCheck)
{
    const int n = d2n.rows;
    const int d = d1n.cols;
    const int rows = bi1 - bi0;

    NnTileResult res;
    res.topIdx.assign(static_cast<size_t>(rows) * kk, -1);
    res.topVal.assign(static_cast<size_t>(rows) * kk, -2.0f);
    if (mutualCheck) {
        res.colBestIdx.assign(static_cast<size_t>(n), -1);
        res.colBestVal.assign(static_cast<size_t>(n), -2.0f);
    }

    cv::Mat simBlock(kNnTileRows, kNnTileCols, CV_32F);   // 1MB 块缓冲，复用

    for (int bj0 = 0; bj0 < n; bj0 += kNnTileCols) {
        const int cols = std::min(kNnTileCols, n - bj0);

        // 块内相似度（SIMD 点积）
        for (int li = 0; li < rows; ++li) {
            const float *a = d1n.ptr<float>(bi0 + li);
            float *srow = simBlock.ptr<float>(li);
            for (int lj = 0; lj < cols; ++lj)
                srow[lj] = nnDotProduct(a, d2n.ptr<float>(bj0 + lj), d);
        }

        // 单次行扫描：列向 top-1 更新 + 行内 top-k 插入归并（跨列块累积）。
        // 注意 colBest 更新必须在 top-k 剪枝之前：被剪枝的低值元素
        // 也可能恰好是某列的全局 top-1
        for (int li = 0; li < rows; ++li) {
            const float *srow = simBlock.ptr<float>(li);
            int *kIdx = &res.topIdx[static_cast<size_t>(li) * kk];
            float *kVal = &res.topVal[static_cast<size_t>(li) * kk];
            const int gi = bi0 + li;
            for (int lj = 0; lj < cols; ++lj) {
                const float v = srow[lj];
                if (mutualCheck && v > res.colBestVal[static_cast<size_t>(bj0 + lj)]) {
                    res.colBestVal[static_cast<size_t>(bj0 + lj)] = v;
                    res.colBestIdx[static_cast<size_t>(bj0 + lj)] = gi;
                }
                if (v <= kVal[kk - 1] && kIdx[kk - 1] != -1)
                    continue;
                int pos = kk - 1;
                while (pos > 0 && kVal[pos - 1] < v) {
                    kVal[pos] = kVal[pos - 1];
                    kIdx[pos] = kIdx[pos - 1];
                    --pos;
                }
                kVal[pos] = v;
                kIdx[pos] = bj0 + lj;
            }
        }
    }
    return res;
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

    cv::Mat d1n, d2n;
    nnNormalizeRows(desc1, d1n);
    nnNormalizeRows(desc2, d2n);

    const int m = d1n.rows, n = d2n.rows;
    const int kk = std::min(std::max(k, 1), n);

    // 行块调度：前 nTiles-1 块提交全局线程池，最后一块当前线程执行——
    // 调用方可能本身是池 worker（管线匹配阶段），避免嵌套等待死锁
    std::vector<int> tileBegin;
    for (int bi0 = 0; bi0 < m; bi0 += kNnTileRows)
        tileBegin.push_back(bi0);
    const int nTiles = static_cast<int>(tileBegin.size());

    std::vector<std::future<NnTileResult>> futures;
    futures.reserve(std::max(0, nTiles - 1));
    for (int t = 0; t < nTiles - 1; ++t) {
        const int bi0 = tileBegin[static_cast<size_t>(t)];
        const int bi1 = std::min(bi0 + kNnTileRows, m);
        futures.push_back(sfm::globalThreadPool().submit(
            nnMatchRowsTile, std::cref(d1n), std::cref(d2n), bi0, bi1, kk, m_mutualCheck));
    }

    // 收集 + 合并列向 top-1（严格大于 → 先合并的行号小的块优先，与全量扫描一致）
    std::vector<int> globalColBestIdx;
    std::vector<float> globalColBestVal;
    if (m_mutualCheck) {
        globalColBestIdx.assign(static_cast<size_t>(n), -1);
        globalColBestVal.assign(static_cast<size_t>(n), -2.0f);
    }

    std::vector<NnTileResult> results;
    results.reserve(static_cast<size_t>(nTiles));
    for (int t = 0; t < nTiles; ++t) {
        if (t < nTiles - 1) {
            results.push_back(futures[static_cast<size_t>(t)].get());
        } else {
            const int bi0 = tileBegin[static_cast<size_t>(t)];
            const int bi1 = std::min(bi0 + kNnTileRows, m);
            results.push_back(nnMatchRowsTile(d1n, d2n, bi0, bi1, kk, m_mutualCheck));
        }
        if (m_mutualCheck) {
            const auto &cbi = results.back().colBestIdx;
            const auto &cbv = results.back().colBestVal;
            for (int j = 0; j < n; ++j) {
                if (cbv[static_cast<size_t>(j)] > globalColBestVal[static_cast<size_t>(j)]) {
                    globalColBestVal[static_cast<size_t>(j)] = cbv[static_cast<size_t>(j)];
                    globalColBestIdx[static_cast<size_t>(j)] = cbi[static_cast<size_t>(j)];
                }
            }
        }
    }

    // mutual 过滤 + 距离阈值 → 输出（与旧实现一致：跳过无匹配行，queryIdx 携带行号）
    outKnnMatches.reserve(static_cast<size_t>(m));
    int globalRow = 0;
    for (const auto &res : results) {
        const int rows = static_cast<int>(res.topIdx.size() / static_cast<size_t>(std::max(kk, 1)));
        for (int li = 0; li < rows; ++li, ++globalRow) {
            const int *kIdx = &res.topIdx[static_cast<size_t>(li) * kk];
            const float *kVal = &res.topVal[static_cast<size_t>(li) * kk];
            if (m_mutualCheck &&
                (kIdx[0] < 0 || globalColBestIdx[static_cast<size_t>(kIdx[0])] != globalRow))
                continue;
            std::vector<cv::DMatch> candidates;
            candidates.reserve(static_cast<size_t>(kk));
            for (int t2 = 0; t2 < kk; ++t2) {
                if (kIdx[t2] < 0)
                    continue;
                const float dist = nnSimToDistance(kVal[t2]);
                if (m_distanceThreshold > 0.0 && dist > m_distanceThreshold)
                    continue;
                candidates.emplace_back(globalRow, kIdx[t2], dist);
            }
            if (!candidates.empty())
                outKnnMatches.push_back(std::move(candidates));
        }
    }
    return !outKnnMatches.empty();
}

bool NearestNeighborMatcherWorker::matchPair(const cv::Mat &desc1, const cv::Mat &desc2,
                                             std::vector<cv::DMatch> &outMatches)
{
    outMatches.clear();
    // top-1 语义与旧实现等价：插入选择 kk=1 保留首个最大值，mutual/阈值同源
    std::vector<std::vector<cv::DMatch>> knn;
    if (!knnMatchPair(desc1, desc2, knn, 1))
        return false;
    for (const auto &row : knn) {
        for (const auto &dm : row)
            outMatches.push_back(dm);
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
