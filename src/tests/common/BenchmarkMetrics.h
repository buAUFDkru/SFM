#ifndef SFM_BENCHMARK_METRICS_H
#define SFM_BENCHMARK_METRICS_H

/**
 * @file BenchmarkMetrics.h
 * @brief 阶段 0：数据集级自比层指标契约与漂移比对（header-only）
 *
 * 自比层五项：注册率 / 点数 / 中位重投影误差 / 各阶段耗时 / 峰值内存。
 * 对标层字段预留但不计算（无真值数据集，见架构 V-02 / O7）。
 */

#include <QDateTime>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace sfm {
namespace bench {

inline constexpr int kSchemaVersion = 1;

struct BenchmarkMetrics {
    QString datasetId;
    QString datasetPath;
    QString timestamp; // ISO 8601

    double registrationRate = 0.0;        // [0, 1]
    int totalPoints = 0;
    double medianReprojectionError = 0.0; // 像素
    double peakMemoryMb = 0.0;
    std::map<std::string, double> stageTimingsMs;

    // 便于排障的附加字段（不参与默认漂移判定）
    int numImages = 0;
    int registeredCameras = 0;
    int matchPairs = 0;
    int mainComponentSize = 0;
};

struct DriftTolerance {
    double absTol = -1.0; // <0 表示不启用绝对容差
    double relTol = -1.0; // <0 表示不启用相对容差；相对 = |cur-base|/max(|base|, eps)
};

struct DriftCheckConfig {
    DriftTolerance registrationRate{0.05, -1.0};
    DriftTolerance totalPoints{-1.0, 0.15};
    DriftTolerance medianReprojectionError{0.5, 0.25};
    DriftTolerance peakMemoryMb{-1.0, 0.30};
    DriftTolerance totalTimingMs{-1.0, 0.50};
};

struct DriftFinding {
    QString metric;
    double baseline = 0.0;
    double current = 0.0;
    double absDelta = 0.0;
    double relDelta = 0.0;
    QString message;
};

struct DriftReport {
    bool ok = true;
    QList<DriftFinding> findings;
};

inline QJsonObject metricsToJson(const BenchmarkMetrics &m)
{
    QJsonObject stageObj;
    for (const auto &kv : m.stageTimingsMs) {
        stageObj.insert(QString::fromStdString(kv.first), kv.second);
    }

    QJsonObject metricsObj;
    metricsObj.insert(QStringLiteral("registrationRate"), m.registrationRate);
    metricsObj.insert(QStringLiteral("totalPoints"), m.totalPoints);
    metricsObj.insert(QStringLiteral("medianReprojectionError"), m.medianReprojectionError);
    metricsObj.insert(QStringLiteral("peakMemoryMb"), m.peakMemoryMb);
    metricsObj.insert(QStringLiteral("stageTimingsMs"), stageObj);

    QJsonObject extras;
    extras.insert(QStringLiteral("numImages"), m.numImages);
    extras.insert(QStringLiteral("registeredCameras"), m.registeredCameras);
    extras.insert(QStringLiteral("matchPairs"), m.matchPairs);
    extras.insert(QStringLiteral("mainComponentSize"), m.mainComponentSize);

    QJsonObject root;
    root.insert(QStringLiteral("schemaVersion"), kSchemaVersion);
    root.insert(QStringLiteral("layer"), QStringLiteral("self"));
    root.insert(QStringLiteral("datasetId"), m.datasetId);
    root.insert(QStringLiteral("datasetPath"), m.datasetPath);
    root.insert(QStringLiteral("timestamp"),
                m.timestamp.isEmpty() ? QDateTime::currentDateTimeUtc().toString(Qt::ISODate)
                                      : m.timestamp);
    root.insert(QStringLiteral("metrics"), metricsObj);
    root.insert(QStringLiteral("extras"), extras);
    // 对标层占位：MVP 不计算
    root.insert(QStringLiteral("benchmarkLayer"), QJsonObject());
    return root;
}

inline bool writeMetricsJson(const BenchmarkMetrics &m, const QString &path, QString *error)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("无法写入指标文件: %1 (%2)")
                         .arg(path, f.errorString());
        }
        return false;
    }
    const QByteArray bytes =
        QJsonDocument(metricsToJson(m)).toJson(QJsonDocument::Indented);
    if (f.write(bytes) != bytes.size()) {
        if (error) {
            *error = QStringLiteral("写入指标文件不完整: %1").arg(path);
        }
        return false;
    }
    return true;
}

inline bool metricsFromJson(const QJsonObject &root, BenchmarkMetrics *out, QString *error)
{
    if (!out) {
        if (error) *error = QStringLiteral("out 为空");
        return false;
    }
    const int ver = root.value(QStringLiteral("schemaVersion")).toInt(0);
    if (ver != kSchemaVersion) {
        if (error) {
            *error = QStringLiteral("不支持的 schemaVersion=%1（期望 %2）")
                         .arg(ver)
                         .arg(kSchemaVersion);
        }
        return false;
    }
    const QJsonObject metricsObj = root.value(QStringLiteral("metrics")).toObject();
    if (metricsObj.isEmpty()) {
        if (error) *error = QStringLiteral("缺少 metrics 对象");
        return false;
    }

    out->datasetId = root.value(QStringLiteral("datasetId")).toString();
    out->datasetPath = root.value(QStringLiteral("datasetPath")).toString();
    out->timestamp = root.value(QStringLiteral("timestamp")).toString();
    out->registrationRate = metricsObj.value(QStringLiteral("registrationRate")).toDouble();
    out->totalPoints = metricsObj.value(QStringLiteral("totalPoints")).toInt();
    out->medianReprojectionError =
        metricsObj.value(QStringLiteral("medianReprojectionError")).toDouble();
    out->peakMemoryMb = metricsObj.value(QStringLiteral("peakMemoryMb")).toDouble();

    out->stageTimingsMs.clear();
    const QJsonObject stageObj = metricsObj.value(QStringLiteral("stageTimingsMs")).toObject();
    for (auto it = stageObj.begin(); it != stageObj.end(); ++it) {
        out->stageTimingsMs[it.key().toStdString()] = it.value().toDouble();
    }

    const QJsonObject extras = root.value(QStringLiteral("extras")).toObject();
    out->numImages = extras.value(QStringLiteral("numImages")).toInt();
    out->registeredCameras = extras.value(QStringLiteral("registeredCameras")).toInt();
    out->matchPairs = extras.value(QStringLiteral("matchPairs")).toInt();
    out->mainComponentSize = extras.value(QStringLiteral("mainComponentSize")).toInt();
    return true;
}

inline bool readMetricsJson(const QString &path, BenchmarkMetrics *out, QString *error)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("无法读取指标文件: %1 (%2)")
                         .arg(path, f.errorString());
        }
        return false;
    }
    QJsonParseError pe;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &pe);
    if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
        if (error) {
            *error = QStringLiteral("指标 JSON 解析失败: %1").arg(pe.errorString());
        }
        return false;
    }
    return metricsFromJson(doc.object(), out, error);
}

inline bool withinTolerance(double baseline, double current, const DriftTolerance &tol,
                            double *absDeltaOut, double *relDeltaOut)
{
    const double absDelta = std::fabs(current - baseline);
    const double denom = std::max(std::fabs(baseline), 1e-9);
    const double relDelta = absDelta / denom;
    if (absDeltaOut) *absDeltaOut = absDelta;
    if (relDeltaOut) *relDeltaOut = relDelta;

    const bool hasAbs = tol.absTol >= 0.0;
    const bool hasRel = tol.relTol >= 0.0;
    if (!hasAbs && !hasRel) {
        return true; // 未配置容差则跳过
    }
    // 任一启用的容差满足即通过（OR）：绝对值小或相对漂移小均可接受
    if (hasAbs && absDelta <= tol.absTol) return true;
    if (hasRel && relDelta <= tol.relTol) return true;
    // 若只启用了一种且未满足，失败；两种都启用且都不满足，失败
    return false;
}

inline void appendFinding(DriftReport *report, const QString &name, double baseline,
                          double current, const DriftTolerance &tol)
{
    double absDelta = 0.0;
    double relDelta = 0.0;
    if (withinTolerance(baseline, current, tol, &absDelta, &relDelta)) {
        return;
    }
    DriftFinding f;
    f.metric = name;
    f.baseline = baseline;
    f.current = current;
    f.absDelta = absDelta;
    f.relDelta = relDelta;
    f.message = QStringLiteral("%1 漂移超容差: baseline=%2 current=%3 |Δ|=%4 rel=%5%")
                    .arg(name)
                    .arg(baseline, 0, 'g', 6)
                    .arg(current, 0, 'g', 6)
                    .arg(absDelta, 0, 'g', 6)
                    .arg(relDelta * 100.0, 0, 'f', 2);
    report->findings.push_back(f);
    report->ok = false;
}

inline DriftReport compareToBaseline(const BenchmarkMetrics &baseline,
                                     const BenchmarkMetrics &current,
                                     const DriftCheckConfig &cfg = DriftCheckConfig())
{
    DriftReport report;
    appendFinding(&report, QStringLiteral("registrationRate"), baseline.registrationRate,
                  current.registrationRate, cfg.registrationRate);
    appendFinding(&report, QStringLiteral("totalPoints"),
                  static_cast<double>(baseline.totalPoints),
                  static_cast<double>(current.totalPoints), cfg.totalPoints);
    appendFinding(&report, QStringLiteral("medianReprojectionError"),
                  baseline.medianReprojectionError, current.medianReprojectionError,
                  cfg.medianReprojectionError);
    appendFinding(&report, QStringLiteral("peakMemoryMb"), baseline.peakMemoryMb,
                  current.peakMemoryMb, cfg.peakMemoryMb);

    const double baseTotal = baseline.stageTimingsMs.count("total")
                                 ? baseline.stageTimingsMs.at("total")
                                 : 0.0;
    const double curTotal = current.stageTimingsMs.count("total")
                                ? current.stageTimingsMs.at("total")
                                : 0.0;
    appendFinding(&report, QStringLiteral("stageTimingsMs.total"), baseTotal, curTotal,
                  cfg.totalTimingMs);
    return report;
}

/// 从误差样本计算中位数；空样本返回 0。
inline double medianOf(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    if (values.size() % 2 == 1) {
        return values[mid];
    }
    const double upper = values[mid];
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                     values.end());
    return 0.5 * (values[mid - 1] + upper);
}

} // namespace bench
} // namespace sfm

#endif // SFM_BENCHMARK_METRICS_H
