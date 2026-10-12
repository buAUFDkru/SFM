#include "pipeline/FeatureExtractionStage.h"

#include "plugin/IDescriptorPlugin.h"
#include "plugin/PluginRegistry.h"
#include "algorithm/ThreadPool.h"

#include <QElapsedTimer>
#include <QFileInfo>

#include <atomic>
#include <future>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace sfm {

bool FeatureExtractionStage::run(const FeatureExtractionInput &in,
                                 std::vector<FeatureResult> *outFeatures,
                                 ThreadPool *threadPool,
                                 CancellationToken *cancel,
                                 const StageLogFn &log,
                                 const StageProgressFn &progress)
{
    if (!outFeatures) return false;

    auto L = [&](const QString &m) { if (log) log(m); };
    auto P = [&](const QString &s, int pct) { if (progress) progress(s, pct); };

    L(QStringLiteral("-------- [阶段 1/6] 特征提取 开始 --------"));
    P(QStringLiteral("特征提取"), 0);
    QElapsedTimer stageTimer;
    stageTimer.start();

    IDescriptorPlugin *plugin =
        PluginRegistry::instance().descriptorPlugin(in.descriptorType);
    if (!plugin) {
        L(QStringLiteral("错误: 描述符插件未加载: %1").arg(in.descriptorType));
        return false;
    }

    const QStringList &files = in.imageFiles;
    const int N = static_cast<int>(files.size());
    outFeatures->clear();
    outFeatures->resize(static_cast<size_t>(N));

    auto extractOne = [&](int idx) -> bool {
        const QString path = files[idx];
        QElapsedTimer imageTimer;
        imageTimer.start();
        if (log) {
            log(QStringLiteral("[特征提取] (%1/%2) 开始: %3")
                    .arg(idx + 1).arg(N).arg(QFileInfo(path).fileName()));
        }
        cv::Mat color = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
        if (color.empty()) {
            if (log) log(QStringLiteral("错误: 无法读取图像: %1").arg(path));
            return false;
        }
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

        ImageEntry entry;
        entry.id = idx;
        entry.filePath = path;
        entry.imageData = color;
        entry.imageGray = gray;

        auto worker = plugin->createWorker(in.descriptorParams);
        if (!worker) {
            if (log) {
                log(QStringLiteral("错误: 描述符插件 createWorker 失败: %1")
                        .arg(in.descriptorType));
            }
            return false;
        }
        FeatureResult fr = worker->extract(entry);
        fr.imageId = idx;
        fr.imageWidth = gray.cols;
        fr.imageHeight = gray.rows;
        fr.imageData = color;
        if (log) {
            log(QStringLiteral("[特征提取] (%1/%2) 完成: %3 关键点, %4x%5, %6 s")
                    .arg(idx + 1).arg(N)
                    .arg(static_cast<int>(fr.keypoints.size()))
                    .arg(gray.cols).arg(gray.rows)
                    .arg(imageTimer.elapsed() / 1000.0, 0, 'f', 1));
        }
        (*outFeatures)[static_cast<size_t>(idx)] = std::move(fr);
        return true;
    };

    if (threadPool && N > 1) {
        std::atomic<bool> allOk{true};
        std::vector<std::future<void>> futures;
        futures.reserve(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) {
            futures.push_back(threadPool->submit(
                [&allOk, &extractOne, i]() {
                    if (!extractOne(i))
                        allOk.store(false, std::memory_order_relaxed);
                }));
        }
        for (auto &f : futures)
            f.get();
        if (!allOk.load(std::memory_order_relaxed))
            return false;
    } else {
        for (int i = 0; i < N; ++i) {
            if (!extractOne(i)) {
                L(QStringLiteral("错误: 特征提取失败"));
                return false;
            }
            if (cancel && cancel->isStopRequested()) return false;
        }
    }

    if (cancel && cancel->isStopRequested()) {
        L(QStringLiteral("[特征提取] 收到停止请求"));
        return false;
    }
    P(QStringLiteral("特征提取"), 100);
    int totalKeypoints = 0;
    for (const auto &f : *outFeatures)
        totalKeypoints += static_cast<int>(f.keypoints.size());
    L(QStringLiteral("[特征提取] 总计: %1 张图像, %2 个关键点 (%3 s)")
          .arg(outFeatures->size()).arg(totalKeypoints)
          .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    L(QStringLiteral("-------- [阶段 1/6] 特征提取 完成 --------"));
    return true;
}

} // namespace sfm
