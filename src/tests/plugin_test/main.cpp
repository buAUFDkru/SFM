#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QLoggingCategory>
#include <QMessageLogContext>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "plugin/IDescriptorPlugin.h"
#include "plugin/IMatcherPlugin.h"
#include "plugin/PluginInfo.h"
#include "plugin/PluginRegistry.h"

namespace {

int failures = 0;

void check(bool cond, const QString &name)
{
    if (cond) {
        qInfo().noquote() << "[PASS]" << name;
    } else {
        qCritical().noquote() << "[FAIL]" << name;
        ++failures;
    }
}

/// 构造一张合成灰度图
ImageEntry makeImage(int id, int rows, int cols)
{
    ImageEntry img;
    img.id = id;
    img.imageGray = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(128));
    img.imageData = img.imageGray.clone();
    return img;
}

/// 用线程池并发调用同一插件的 createWorker() + extract()，断言与串行一致
/// （验证"工厂无状态 + Worker 逐任务独占"在多线程下不会破坏算法状态）
void runConcurrencyCheck(IDescriptorPlugin *plugin)
{
    const int kThreads = 8;
    const int kTasksPerThread = 20;

    const QJsonObject params{{"scale", 1.0}};
    const ImageEntry refImage = makeImage(0, 240, 320);

    // 串行参考结果
    auto refWorker = plugin->createWorker(params);
    const FeatureResult ref = refWorker->extract(refImage);

    std::atomic<int> mismatch{0};

    auto workerFn = [&]() {
        for (int k = 0; k < kTasksPerThread; ++k) {
            auto w = plugin->createWorker(params);
            if (!w) {
                ++mismatch;
                continue;
            }
            const FeatureResult r = w->extract(refImage);
            const double descSum =
                r.descriptors.empty() ? 0.0 : cv::sum(r.descriptors).val[0];
            const double refSum =
                ref.descriptors.empty() ? 0.0 : cv::sum(ref.descriptors).val[0];
            if (r.keypoints.size() != ref.keypoints.size() || descSum != refSum)
                ++mismatch;
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back(workerFn);
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }

    const int total = kThreads * kTasksPerThread;
    check(mismatch.load() == 0,
          QStringLiteral("并发一致性 (%1 线程 × %2 任务, 0 不一致)").arg(kThreads).arg(kTasksPerThread)
              + (mismatch.load() == 0 ? QStringLiteral(" — 全部与串行一致")
                                      : QStringLiteral(" — 不一致 %1 处").arg(mismatch.load())));
}

/// 构造一张固定种子的随机噪声构图 (256x256 CV_8UC1)，保证 SIFT/KAZE 有丰富特征
ImageEntry makeSyntheticTextureImage(int id)
{
    ImageEntry img;
    img.id = id;
    img.imageGray = cv::Mat(256, 256, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(img.imageGray, cv::RNG::UNIFORM, 0, 255);
    img.imageData = img.imageGray.clone();
    return img;
}

/// 构造一对固定种子的随机浮点描述符矩阵 (desc2 = desc1 深度复制，最近邻距离≈0)
void makeSyntheticFeatures(cv::Mat &desc1, cv::Mat &desc2)
{
    desc1 = cv::Mat(32, 64, CV_32F);
    cv::RNG rng(7);
    rng.fill(desc1, cv::RNG::UNIFORM, 0.f, 1.f);
    desc2 = desc1.clone();
}

/// 对描述符插件做 P2 并发一致性检查（8 线程 × 10 任务，各任务内 createWorker + extract）
void runDescriptorConcurrencyCheck(IDescriptorPlugin *plugin)
{
    const int kThreads = 8;
    const int kTasksPerThread = 10;

    const ImageEntry img = makeSyntheticTextureImage(0);

    // 串行参考结果
    auto refWorker = plugin->createWorker(QJsonObject{});
    if (!refWorker)
        return;
    const FeatureResult ref = refWorker->extract(img);

    std::atomic<int> mismatch{0};

    auto workerFn = [&]() {
        for (int k = 0; k < kTasksPerThread; ++k) {
            auto w = plugin->createWorker(QJsonObject{});
            if (!w) {
                ++mismatch;
                continue;
            }
            const FeatureResult r = w->extract(img);
            const double descSum =
                r.descriptors.empty() ? 0.0 : cv::sum(r.descriptors).val[0];
            const double refSum =
                ref.descriptors.empty() ? 0.0 : cv::sum(ref.descriptors).val[0];
            if (r.keypoints.size() != ref.keypoints.size() || descSum != refSum)
                ++mismatch;
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back(workerFn);
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }

    const int total = kThreads * kTasksPerThread;
    check(mismatch.load() == 0,
          QStringLiteral("并发一致性 (%1 线程 × %2 任务, 0 不一致)").arg(kThreads).arg(kTasksPerThread)
              + (mismatch.load() == 0 ? QStringLiteral(" — 全部与串行一致")
                                      : QStringLiteral(" — 不一致 %1 处").arg(mismatch.load())));
}

int run(const QString &pluginsDir)
{
    auto &reg = PluginRegistry::instance();

    const int loaded = reg.scanPlugins(pluginsDir);
    check(loaded >= 1, QStringLiteral("扫描到插件(%1)").arg(loaded));

    const auto infos = reg.plugins();
    check(!infos.isEmpty(), "plugins() 非空");

    auto *plugin = reg.descriptorPlugin("SampleDescriptor");
    check(plugin != nullptr, "按 id 加载 SampleDescriptor 插件");
    if (!plugin) {
        qCritical() << "  请确认插件 DLL 已部署到:" << pluginsDir;
        return failures > 0 ? 1 : 0;
    }

    check(plugin->version() == "1.0.0", "版本号正确读取");

    const auto schema = plugin->paramSchema();
    check(schema.size() == 1 && schema[0].name == "scale", "参数协议(scale)正确读取");

    QString err;
    check(plugin->validateParams(QJsonObject{{"scale", 1.0}}, &err),
          "合法参数校验通过");
    check(!plugin->validateParams(QJsonObject{{"scale", "bad"}}, &err),
          "非法参数被拒绝: " + err);

    // 工厂→Worker 的基本调用
    {
        auto w = plugin->createWorker(QJsonObject{{"scale", 1.0}});
        check(w != nullptr, "createWorker 创建成功");
        if (w) {
            const FeatureResult r = w->extract(makeImage(1, 240, 320));
            check(r.imageId == 1 && r.keypoints.size() == 24,
                  QStringLiteral("extract 确定性输出 (keypoints=%1)").arg(r.keypoints.size()));
        }
    }

    // 线程池并发一致性
    runConcurrencyCheck(plugin);

    // ── P2 描述符插件 (SIFT / KAZE) ──
    for (const auto &descName : { QStringLiteral("SIFT"), QStringLiteral("KAZE") }) {
        auto *dp = reg.descriptorPlugin(descName);
        check(dp != nullptr, "按 id 加载 " + descName + " 插件");
        if (!dp)
            continue;
        auto dw = dp->createWorker(QJsonObject{});
        check(dw != nullptr, descName + " createWorker 创建成功");
        if (dw) {
            const FeatureResult r = dw->extract(makeSyntheticTextureImage(1));
            check(r.keypoints.size() > 10,
                  QStringLiteral("%1 提取合成纹理特征 keypoints=%2 (>10)")
                      .arg(descName)
                      .arg(r.keypoints.size()));
        }
        runDescriptorConcurrencyCheck(dp);
    }

    // ── P2 匹配器插件 (FLANN / BruteForce) ──
    for (const auto &matcherName : { QStringLiteral("FLANN"), QStringLiteral("BruteForce") }) {
        auto *mp = reg.matcherPlugin(matcherName);
        check(mp != nullptr, "按 id 加载 " + matcherName + " 插件");
        if (!mp)
            continue;
        auto mw = mp->createWorker(QJsonObject{});
        check(mw != nullptr, matcherName + " createWorker 创建成功");
        if (mw) {
            cv::Mat desc1, desc2;
            makeSyntheticFeatures(desc1, desc2);

            std::vector<std::vector<cv::DMatch>> knn;
            const bool knnOk = mw->knnMatchPair(desc1, desc2, knn, 2);
            int good = 0;
            for (const auto &pair : knn) {
                if (pair.size() >= 2 && pair[0].distance < 0.8 * pair[1].distance)
                    ++good;
            }
            check(knnOk && good > 25,
                  QStringLiteral("%1 knnMatch 成功过滤后匹配数 %2 (>25, 距离≈0)")
                      .arg(matcherName)
                      .arg(good));

            std::vector<cv::DMatch> matches;
            const bool singleOk = mw->matchPair(desc1, desc2, matches);
            double maxD = 0.0;
            for (const auto &m : matches)
                maxD = std::max(maxD, static_cast<double>(m.distance));
            check(singleOk && matches.size() == 32 && maxD < 1e-6,
                  QStringLiteral("%1 matchPair 返回 %2 个匹配 (距离<1e-6)")
                      .arg(matcherName)
                      .arg(matches.size()));
        }
    }

    return failures > 0 ? 1 : 0;
}

} // namespace

int main(int argc, char *argv[])
{
    qInstallMessageHandler([](QtMsgType t, const QMessageLogContext &, const QString &msg) {
        std::fprintf(stderr, "[qt:%d] %s\n", static_cast<int>(t), msg.toLocal8Bit().constData());
        std::fflush(stderr);
    });

    QCoreApplication app(argc, argv);

    QString pluginsDir;
    if (argc > 1) {
        pluginsDir = QString::fromLocal8Bit(argv[1]);
    } else {
        pluginsDir = QCoreApplication::applicationDirPath() + QStringLiteral("/../../../app/plugins");
    }

    const int result = run(pluginsDir);

    std::fprintf(stderr, "%% RESULT %d\n", result);
    std::fprintf(stderr, "%% PLUGINS_DIR %s\n", pluginsDir.toLocal8Bit().constData());

    if (result == 0)
        qInfo().noquote() << "=== ALL TESTS PASSED ===";
    else
        qCritical().noquote() << "=== TESTS FAILED ===";

    return result;
}