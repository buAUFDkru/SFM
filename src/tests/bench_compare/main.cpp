/**
 * @file main.cpp
 * @brief 阶段 0：BenchmarkMetrics JSON 契约与相对基线漂移告警
 *
 * 两种模式：
 *  1) 无额外参数 → QtTest 单元测试（fixture 驱动）
 *  2) --baseline <a.json> --current <b.json> [--fail-on-drift]
 *     → CI 秒级漂移比对（默认仅告警 exit 0；加 --fail-on-drift 则超容差 exit 1）
 */

#include <QtTest/QtTest>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <cstdio>

#include "BenchmarkMetrics.h"

#ifndef BENCH_FIXTURES_DIR
#define BENCH_FIXTURES_DIR "."
#endif

namespace {

QString argValue(const QStringList &args, const QString &key)
{
    const QString prefix = key + QLatin1Char('=');
    for (int i = 0; i < args.size(); ++i) {
        if (args[i] == key) {
            if (i + 1 < args.size()) return args[i + 1];
            return {};
        }
        if (args[i].startsWith(prefix)) {
            return args[i].mid(prefix.size());
        }
    }
    return {};
}

bool hasFlag(const QStringList &args, const QString &flag)
{
    return args.contains(flag);
}

int runCliCompare(const QStringList &args)
{
    const QString baselinePath = argValue(args, QStringLiteral("--baseline"));
    const QString currentPath = argValue(args, QStringLiteral("--current"));
    const bool failOnDrift = hasFlag(args, QStringLiteral("--fail-on-drift"));

    if (baselinePath.isEmpty() || currentPath.isEmpty()) {
        std::printf("用法: bench_compare --baseline <基线.json> --current <当前.json> "
                    "[--fail-on-drift]\n");
        return 2;
    }

    sfm::bench::BenchmarkMetrics baseline;
    sfm::bench::BenchmarkMetrics current;
    QString err;
    if (!sfm::bench::readMetricsJson(baselinePath, &baseline, &err)) {
        std::printf("错误: %s\n", qPrintable(err));
        return 2;
    }
    if (!sfm::bench::readMetricsJson(currentPath, &current, &err)) {
        std::printf("错误: %s\n", qPrintable(err));
        return 2;
    }

    const sfm::bench::DriftReport report = sfm::bench::compareToBaseline(baseline, current);
    if (report.ok) {
        std::printf("[漂移告警] 通过: dataset=%s 相对基线无超容差漂移\n",
                    qPrintable(current.datasetId));
        return 0;
    }

    std::printf("[漂移告警] 检测到 %d 项超容差:\n",
                static_cast<int>(report.findings.size()));
    for (const auto &f : report.findings) {
        std::printf("  - %s\n", qPrintable(f.message));
    }
    if (failOnDrift) {
        std::printf("[漂移告警] --fail-on-drift 已开启，返回失败\n");
        return 1;
    }
    std::printf("[漂移告警] 仅告警模式（默认），返回成功；下一迭代可加 --fail-on-drift 收紧\n");
    return 0;
}

} // namespace

class BenchCompareTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void roundTripWriteRead();
    void compareWithinTolerance();
    void compareDetectsDrift();
    void rejectBadSchemaVersion();
    void medianHelper();

private:
    QString fixturePath(const char *name) const;
};

void BenchCompareTest::initTestCase()
{
    QVERIFY2(QDir(QString::fromUtf8(BENCH_FIXTURES_DIR)).exists(),
             "fixtures 目录不存在，检查 bench_compare.pro 的 BENCH_FIXTURES_DIR");
}

QString BenchCompareTest::fixturePath(const char *name) const
{
    return QDir(QString::fromUtf8(BENCH_FIXTURES_DIR)).absoluteFilePath(QString::fromUtf8(name));
}

void BenchCompareTest::roundTripWriteRead()
{
    sfm::bench::BenchmarkMetrics m;
    m.datasetId = QStringLiteral("roundtrip");
    m.datasetPath = QStringLiteral("D:/tmp/ds");
    m.registrationRate = 0.875;
    m.totalPoints = 1234;
    m.medianReprojectionError = 0.55;
    m.peakMemoryMb = 256.5;
    m.stageTimingsMs["feature_extract"] = 10.0;
    m.stageTimingsMs["total"] = 99.0;
    m.numImages = 8;
    m.registeredCameras = 7;

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = dir.filePath(QStringLiteral("metrics.json"));

    QString err;
    QVERIFY2(sfm::bench::writeMetricsJson(m, path, &err), qPrintable(err));

    sfm::bench::BenchmarkMetrics loaded;
    QVERIFY2(sfm::bench::readMetricsJson(path, &loaded, &err), qPrintable(err));
    QCOMPARE(loaded.datasetId, m.datasetId);
    QCOMPARE(loaded.totalPoints, m.totalPoints);
    QCOMPARE(loaded.registrationRate, m.registrationRate);
    QCOMPARE(loaded.medianReprojectionError, m.medianReprojectionError);
    QCOMPARE(loaded.peakMemoryMb, m.peakMemoryMb);
    QCOMPARE(loaded.numImages, m.numImages);
    QCOMPARE(loaded.registeredCameras, m.registeredCameras);
    QVERIFY(loaded.stageTimingsMs.count("total") == 1);
    QCOMPARE(loaded.stageTimingsMs.at("total"), 99.0);
}

void BenchCompareTest::compareWithinTolerance()
{
    sfm::bench::BenchmarkMetrics baseline;
    sfm::bench::BenchmarkMetrics current;
    QString err;
    QVERIFY2(sfm::bench::readMetricsJson(fixturePath("baseline.json"), &baseline, &err),
             qPrintable(err));
    QVERIFY2(sfm::bench::readMetricsJson(fixturePath("current_ok.json"), &current, &err),
             qPrintable(err));

    const sfm::bench::DriftReport report = sfm::bench::compareToBaseline(baseline, current);
    QVERIFY2(report.ok, "容差内漂移应判定通过");
    QCOMPARE(report.findings.size(), 0);
}

void BenchCompareTest::compareDetectsDrift()
{
    sfm::bench::BenchmarkMetrics baseline;
    sfm::bench::BenchmarkMetrics current;
    QString err;
    QVERIFY2(sfm::bench::readMetricsJson(fixturePath("baseline.json"), &baseline, &err),
             qPrintable(err));
    QVERIFY2(sfm::bench::readMetricsJson(fixturePath("current_drift.json"), &current, &err),
             qPrintable(err));

    const sfm::bench::DriftReport report = sfm::bench::compareToBaseline(baseline, current);
    QVERIFY2(!report.ok, "超容差应判定失败");
    QVERIFY2(!report.findings.isEmpty(), "应至少有一条 finding");

    bool sawRegistration = false;
    for (const auto &f : report.findings) {
        if (f.metric == QStringLiteral("registrationRate")) {
            sawRegistration = true;
            break;
        }
    }
    QVERIFY2(sawRegistration, "应报告 registrationRate 漂移");
}

void BenchCompareTest::rejectBadSchemaVersion()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = dir.filePath(QStringLiteral("bad.json"));
    {
        QJsonObject root;
        root.insert(QStringLiteral("schemaVersion"), 99);
        root.insert(QStringLiteral("metrics"), QJsonObject());
        QFile f(path);
        QVERIFY(f.open(QIODevice::WriteOnly | QIODevice::Truncate));
        f.write(QJsonDocument(root).toJson());
    }

    sfm::bench::BenchmarkMetrics m;
    QString err;
    QVERIFY(!sfm::bench::readMetricsJson(path, &m, &err));
    QVERIFY(err.contains(QStringLiteral("schemaVersion")));
}

void BenchCompareTest::medianHelper()
{
    QCOMPARE(sfm::bench::medianOf({}), 0.0);
    QCOMPARE(sfm::bench::medianOf({3.0}), 3.0);
    QCOMPARE(sfm::bench::medianOf({1.0, 3.0, 2.0}), 2.0);
    QCOMPARE(sfm::bench::medianOf({1.0, 2.0, 3.0, 4.0}), 2.5);
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    const QStringList args = QCoreApplication::arguments();
    if (hasFlag(args, QStringLiteral("--baseline")) ||
        !argValue(args, QStringLiteral("--baseline")).isEmpty()) {
        return runCliCompare(args);
    }

    BenchCompareTest tc;
    return QTest::qExec(&tc, argc, argv);
}

#include "main.moc"
