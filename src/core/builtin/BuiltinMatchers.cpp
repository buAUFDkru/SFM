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
    normType.defaultValue = -1;
    normType.min = -1;
    normType.max = 5;
    normType.group = QStringLiteral("BruteForce");
    normType.tooltip = QStringLiteral("距离度量：-1=按描述符类型自动选择(二进制→Hamming，浮点→L2)；可显式指定 0=NORM_INF 1=NORM_L1 2=NORM_L2 4=NORM_L2SQR 5=NORM_HAMMING 6=NORM_HAMMING2");

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
