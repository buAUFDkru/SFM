#include "ReconstructionPipeline.h"

#include <cstdio>

#include "algorithm/Initialization.h"
#include "algorithm/MatcherService.h"
#include "algorithm/PointCloudBuilder.h"
#include "algorithm/SfmLogger.h"
#include "algorithm/ThreadPool.h"
#include "algorithm/Triangulate.h"
#include "pipeline/DenseReconstructionFactory.h"
#include "pipeline/IDenseReconstructor.h"
#include "plugin/IDescriptorPlugin.h"
#include "plugin/PluginRegistry.h"

#include <QCoreApplication>
#include <QDebug>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>

ReconstructionPipeline::ReconstructionPipeline()
{
    // 登记内置密集重建方法（StereoSGBM / PatchMatch），幂等；
    // 静态库链接下实现 TU 的静态注册对象不会自动拉入，必须在运行前显式注册。
    DenseReconstructionFactory::instance().registerBuiltinMethods();
}

ReconstructionPipeline::~ReconstructionPipeline()
{
    // 清理 SfmLogger 回调并卸载 Qt 消息处理器，避免静态单例持有已销毁对象的引用。
    // 注意：这里假设 ReconstructionPipeline 是当前 SfmLogger 回调的注册者。
    SfmLogger::instance().uninstallAsMessageHandler();
    SfmLogger::instance().clearCallbacks();
}

void ReconstructionPipeline::setCallbacks(IReconstructionCallbacks *callbacks)
{
    m_callbacks = callbacks;

    // 将 ReconstructionPipeline 的日志回调注册到全局 SfmLogger。
    // 这样项目中所有 qDebug/qWarning 以及通过 SfmLogger 主动输出的日志
    // 都能通过 IReconstructionCallbacks::onLog() 实时转发到 UI。
    if (m_callbacks) {
        SfmLogger::instance().setDefaultCallback(
            [this](const QString &msg) { log(msg); });
        SfmLogger::instance().installAsMessageHandler();
    } else {
        SfmLogger::instance().uninstallAsMessageHandler();
        SfmLogger::instance().clearCallbacks();
    }
}

void ReconstructionPipeline::waitForDenseReconstructionMethod()
{
    std::unique_lock<std::mutex> lock(m_denseMethodMutex);
    m_denseMethodCv.wait(lock, [this] {
        return m_denseMethodReady || m_stopRequested;
    });
}

void ReconstructionPipeline::setDenseReconstructionMethod(
    const QString &method,
    const QMap<QString, QVariant> &params)
{
    {
        std::lock_guard<std::mutex> lock(m_denseMethodMutex);
        m_config.denseReconstructionMethod = method;
        m_config.denseReconstructionParams = params;
        m_denseMethodReady = true;
    }
    m_denseMethodCv.notify_one();
}

void ReconstructionPipeline::requestStop()
{
    m_stopRequested = true;
    // 如果正在等待 UI 选择密集重建方法，唤醒等待以避免死锁
    {
        std::lock_guard<std::mutex> lock(m_denseMethodMutex);
        m_denseMethodReady = true;
    }
    m_denseMethodCv.notify_one();
}

void ReconstructionPipeline::setConfig(const ReconstructionConfig &config)
{
    m_config = config;
}

ReconstructionConfig ReconstructionPipeline::config() const
{
    return m_config;
}

bool ReconstructionPipeline::extractFeaturesViaPlugins(sfm::ThreadPool *threadPool)
{
    IDescriptorPlugin *plugin =
        PluginRegistry::instance().descriptorPlugin(m_config.descriptorType);
    if (!plugin) {
        log(QString("错误: 描述符插件未加载: %1").arg(m_config.descriptorType));
        return false;
    }

    const QStringList &files = m_config.imageFiles;
    const int N = static_cast<int>(files.size());
    m_features.clear();
    m_features.resize(N);

    auto extractOne = [this, plugin](int idx) -> bool {
        const QString path = m_config.imageFiles[idx];
        cv::Mat color = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
        if (color.empty()) {
            log(QString("错误: 无法读取图像: %1").arg(path));
            return false;
        }
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

        ImageEntry entry;
        entry.id = idx;
        entry.filePath = path;
        entry.imageData = color;
        entry.imageGray = gray;

        // 逐任务独占 Worker（工厂无状态 + Worker 单线程契约）
        auto worker = plugin->createWorker(m_config.descriptorParams);
        if (!worker) {
            log(QString("错误: 描述符插件 createWorker 失败: %1")
                    .arg(m_config.descriptorType));
            return false;
        }
        FeatureResult fr = worker->extract(entry);
        fr.imageId = idx;
        fr.imageWidth = gray.cols;
        fr.imageHeight = gray.rows;
        fr.imageData = color;
        m_features[idx] = std::move(fr);
        return true;
    };

    if (threadPool && N > 1) {
        std::atomic<bool> allOk{true};
        std::vector<std::future<void>> futures;
        futures.reserve(N);
        for (int i = 0; i < N; ++i) {
            futures.push_back(threadPool->submit(
                [&allOk, extractOne, i]() {
                    if (!extractOne(i)) {
                        allOk.store(false, std::memory_order_relaxed);
                    }
                }));
        }
        for (auto &f : futures) {
            f.get();
        }
        return allOk.load(std::memory_order_relaxed);
    }

    // 串行回退
    for (int i = 0; i < N; ++i) {
        if (!extractOne(i)) {
            log("错误: 特征提取失败");
            return false;
        }
        if (m_stopRequested) return false;
    }
    return true;
}

ReconstructionOutput ReconstructionPipeline::execute()
{
    m_stopRequested = false;
    ReconstructionOutput output;

    if (m_config.imageFiles.isEmpty()) {
        log("错误: 未加载图像");
        return output;
    }

    log("========== SFM 重建流水线启动 ==========");
    log(QString("[配置] 图像数量: %1").arg(m_config.imageFiles.size()));
    m_config.imageFiles.sort();

    // 关闭 OpenCV 内部多线程，避免与 sfm3.0 线程池/插件逐任务 Worker 模型叠加
    cv::setNumThreads(1);

    // 创建 CPU 密集型线程池。
    // threadCount 为 0 时默认使用逻辑核心数 / 4，保留核心给 UI/OS（4 核 8 线程 → 2 线程）。
    sfm::ThreadPool threadPool(sfm::make_cpu_pool_options(
        static_cast<std::size_t>(m_config.threadCount)));
    log(QString("[线程池] 工作线程数: %1").arg(threadPool.thread_count()));

    QElapsedTimer stageTimer;
    QElapsedTimer totalTimer;
    totalTimer.start();

    // =====================================================================
    // Stage 1/6: 特征提取（插件）
    // =====================================================================
    log("-------- [阶段 1/6] 特征提取 开始 --------");
    setProgress("特征提取", 0);
    stageTimer.start();

    if (!extractFeaturesViaPlugins(&threadPool)) {
        log("错误: 特征提取失败");
        return output;
    }

    if (m_stopRequested) { log("[特征提取] 收到停止请求"); return output; }
    setProgress("特征提取", 100);

    int totalKeypoints = 0;
    for (size_t i = 0; i < m_features.size(); ++i) {
        totalKeypoints += static_cast<int>(m_features[i].keypoints.size());
    }
    log(QString("[特征提取] 总计: %1 张图像, %2 个关键点 (%3 s)")
        .arg(m_features.size()).arg(totalKeypoints)
        .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    log("-------- [阶段 1/6] 特征提取 完成 --------");

    // =====================================================================
    // Stage 2/6: 特征匹配（MatcherService 编排）
    // =====================================================================
    log("-------- [阶段 2/6] 特征匹配 开始 --------");
    setProgress("特征匹配", 0);
    stageTimer.restart();

    MatcherService matcherService(&PluginRegistry::instance());
    std::vector<MatchPair> matchPairs;
    if (!matcherService.matchFeatures(
            m_features, m_config.matcherType, m_config.matcherParams, matchPairs,
            &threadPool, [this](const QString &msg) { log(msg); })) {
        log("错误: 特征匹配失败");
        return output;
    }

    if (m_stopRequested) { log("[特征匹配] 收到停止请求"); return output; }
    setProgress("特征匹配", 100);

    log(QString("[特征匹配] 总计: %1 对图像连接 (%2 s)")
        .arg(matchPairs.size())
        .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    log("-------- [阶段 2/6] 特征匹配 完成 --------");

    cv::Mat K = m_config.cameraMatrix.empty()
                ? Triangulate::instance()->estimateK(m_features)
                : m_config.cameraMatrix.clone();
    if (!m_config.cameraMatrix.empty()) {
        log(QString("[内参矩阵] 使用外部提供的K: fx=%1 fy=%2 cx=%3 cy=%4")
            .arg(K.at<double>(0, 0), 0, 'f', 2)
            .arg(K.at<double>(1, 1), 0, 'f', 2)
            .arg(K.at<double>(0, 2), 0, 'f', 2)
            .arg(K.at<double>(1, 2), 0, 'f', 2));
    } else {
        log(QString("[内参矩阵] 使用自动估计的K: fx=%1 fy=%2 cx=%3 cy=%4")
            .arg(K.at<double>(0, 0), 0, 'f', 2)
            .arg(K.at<double>(1, 1), 0, 'f', 2)
            .arg(K.at<double>(0, 2), 0, 'f', 2)
            .arg(K.at<double>(1, 2), 0, 'f', 2));
    }
    InitResult initResult = Initialization::instance()->findInitialPair(m_features, matchPairs, K);
    if (initResult.score < 0) {
        log("错误: 未找到合适的初始化图像对，终止重建");
        return output;
    }
    log(QString("[初始化] 种子对: 图像 #%1 ↔ #%2, 评分=%3")
        .arg(initResult.imageId1).arg(initResult.imageId2)
        .arg(initResult.score, 0, 'f', 4));

    ViewGraph viewGraph = ViewGraph::buildFromPairs(matchPairs, static_cast<int>(m_features.size()));

    // =====================================================================
    // Stage 3/6: 稀疏重建
    // =====================================================================
    log("-------- [阶段 3/6] 稀疏重建 开始 --------");
    setProgress("稀疏重建", 0);
    stageTimer.restart();

    std::vector<Point3D> sparsePoints;
    std::vector<CameraPose> cameraPoses;
    if (!Triangulate::instance()->triangulate(
            matchPairs, m_features, initResult, viewGraph, sparsePoints, cameraPoses,
            m_config.fixCameraMatrix)) {
        log("错误: 三角测量失败");
        return output;
    }

    if (m_stopRequested) { log("[稀疏重建] 收到停止请求"); return output; }
    setProgress("稀疏重建", 100);

    log(QString("[稀疏重建] 生成: %1 个三维点, %2 个相机位姿 (%3 s)")
        .arg(sparsePoints.size()).arg(cameraPoses.size())
        .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    log("-------- [阶段 3/6] 稀疏重建 完成 --------");

    // =====================================================================
    // Stage 4/6: 点云优化
    // =====================================================================
    log("-------- [阶段 4/6] 点云优化 开始 --------");
    setProgress("点云优化", 0);
    stageTimer.restart();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    PointCloudBuilder::instance()->convertToPCL(sparsePoints, cloud);
    PointCloudBuilder::instance()->optimizeProcess(cloud);

    std::vector<Point3D> optPoints;
    optPoints.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
        Point3D p;
        p.id = static_cast<int>(optPoints.size());
        p.position = cv::Point3d(pt.x, pt.y, pt.z);
        p.color = cv::Vec3b(pt.b, pt.g, pt.r);
        optPoints.push_back(p);
    }
    setProgress("点云优化", 100);
    log(QString("[点云优化] 完成: %1 个点 (%2 s)")
        .arg(optPoints.size())
        .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    log("-------- [阶段 4/6] 点云优化 完成 --------");

    // 发射稀疏点云供UI显示
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sparsePCL(new pcl::PointCloud<pcl::PointXYZRGB>);
        PointCloudBuilder::instance()->convertToPCL(optPoints, sparsePCL);
        // 坐标系转换：OpenCV y-down → 标准y-up（绕X轴旋转180°）
        for (auto &pt : sparsePCL->points) { pt.y = -pt.y; pt.z = -pt.z; }
        emitStageCloud(sparsePCL, "sparse");
        // 保存稀疏点云到项目目录（y-up 坐标系）
        saveStageCloud(sparsePCL, resolveProjectFilePath("sparse_cloud.ply"), "稀疏重建");
    }

    // 保存稀疏重建结果，供后续单独执行密集重建
    m_sparseReady = true;
    m_cameraPoses = cameraPoses;
    m_matchPairs = matchPairs;
    m_K = K.clone();
    m_optPoints = optPoints;

    // =====================================================================
    // Stage 5-6: 密集重建、点云生成（方法实现于 P4，未注册时跳过）
    // =====================================================================
    runDenseStages(output);

    if (m_stopRequested) { log("[流水线] 收到停止请求"); return output; }

    output.totalElapsedMs = totalTimer.elapsed();

    log(QString("[输出] 稀疏点云: %1 点").arg(m_optPoints.size()));

    setProgress("完成", 100);
    log(QString("========== 重建流水线完成 (总耗时 %1 s) ==========")
        .arg(totalTimer.elapsed() / 1000.0, 0, 'f', 1));

    m_lastOutput = output;
    return output;
}

void ReconstructionPipeline::runDenseStages(ReconstructionOutput &output)
{
    if (!m_sparseReady) {
        log("[密集重建] 错误：未执行稀疏重建");
        return;
    }

    QElapsedTimer stageTimer;

    // =====================================================================
    // Stage 5/6: 密集重建方法选择与执行
    // =====================================================================
    log("-------- [阶段 5/6] 密集重建 开始 --------");
    setProgress("密集重建", 0);
    stageTimer.restart();

    // 若配置中未指定密集重建方法，则请求 UI 选择
    if (m_config.denseReconstructionMethod.isEmpty()) {
        log("[密集重建] 等待选择密集重建方法...");
        if (m_callbacks) {
            m_callbacks->onDenseReconstructionMethodRequested();
        }
        waitForDenseReconstructionMethod();
    }

    if (m_stopRequested) { log("[密集重建] 收到停止请求"); return; }

    std::vector<StereoPairResult> stereoResults;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    try {
        // 通过工厂创建指定的密集重建方法
        auto reconstructor = DenseReconstructionFactory::instance()
                                 .create(m_config.denseReconstructionMethod);
        if (!reconstructor) {
            log(QString("[密集重建] 未找到方法: %1，跳过密集重建（方法实现于 P4）")
                    .arg(m_config.denseReconstructionMethod));
        } else {
            reconstructor->setConfig(m_config.denseReconstructionParams);

            // 设置日志回调，让密集重建方法能实时输出进度日志到 UI
            reconstructor->setLogCallback(
                [this](const QString &msg) { log(msg); });

            DenseReconstructionInput denseInput;
            denseInput.cameraPoses = m_cameraPoses;
            denseInput.matchPairs = m_matchPairs;
            denseInput.features = m_features;
            denseInput.cameraMatrix = m_K;
            denseInput.imageFiles = m_config.imageFiles;
            denseInput.projectPath = m_config.projectPath;

            log(QString("[密集重建] 使用方法: %1").arg(reconstructor->displayName()));
            DenseReconstructionOutput denseOutput = reconstructor->reconstruct(denseInput);

            stereoResults = std::move(denseOutput.stereoResults);
            denseCloud = std::move(denseOutput.denseCloud);

            if (!denseOutput.success) {
                log(QString("[密集重建] 失败: %1").arg(denseOutput.errorMessage));
            } else {
                log(QString("[密集重建] 内部完成, 密集点云=%1 点")
                        .arg(denseCloud ? static_cast<int>(denseCloud->size()) : 0));
            }
        }

        if (m_stopRequested) { log("[密集重建] 收到停止请求"); return; }
        setProgress("密集重建", 100);

        int validDepthMaps = 0;
        for (const auto &r : stereoResults) {
            if (!r.depth.empty()) validDepthMaps++;
        }
        log(QString("[密集重建] 完成: %1/%2 对生成有效深度图, 密集点云 %3 点 (%4 s)")
            .arg(validDepthMaps).arg(stereoResults.size())
            .arg(denseCloud ? static_cast<int>(denseCloud->size()) : 0)
            .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    } catch (const std::exception &e) {
        log(QString("[密集重建] 异常: %1，跳过密集重建").arg(e.what()));
    } catch (...) {
        log("[密集重建] 未知异常，跳过密集重建");
    }
    log("-------- [阶段 5/6] 密集重建 完成 --------");

    // =====================================================================
    // Stage 6/6: 密集点云优化与显示
    // =====================================================================
    log("-------- [阶段 6/6] 密集点云生成 开始 --------");
    setProgress("密集点云生成", 0);
    stageTimer.restart();

    try {
        int totalPoints = 0;
        if (denseCloud && !denseCloud->empty()) {
            totalPoints = static_cast<int>(denseCloud->size());

            // 密集点云优化（离群去除 + 体素下采样 + 法向量估计 + 离群去除）
            PointCloudBuilder::instance()->optimizeProcessDense(denseCloud);
        }

        log(QString("[密集点云] 反投影完成, 总点数=%1").arg(totalPoints));

        if (m_stopRequested) { log("[密集点云生成] 收到停止请求"); return; }
        setProgress("密集点云生成", 100);

        log(QString("[密集点云生成] 完成: %1 个点 (%2 s)")
            .arg(static_cast<int>(denseCloud->size()))
            .arg(stageTimer.elapsed() / 1000.0, 0, 'f', 1));
    } catch (const std::exception &e) {
        log(QString("[密集点云生成] 异常: %1").arg(e.what()));
    } catch (...) {
        log("[密集点云生成] 未知异常");
    }
    log("-------- [阶段 6/6] 密集点云生成 完成 --------");

    // 发射密集点云供UI显示
    if (denseCloud && !denseCloud->empty()) {
        // 坐标系转换：OpenCV y-down → 标准y-up（绕X轴旋转180°）
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseDisplay(new pcl::PointCloud<pcl::PointXYZRGB>);

        denseDisplay->resize(denseCloud->size());
        for (size_t i = 0; i < denseCloud->size(); ++i) {
            denseDisplay->points[i] = denseCloud->points[i];
            denseDisplay->points[i].y = -denseDisplay->points[i].y;
            denseDisplay->points[i].z = -denseDisplay->points[i].z;
        }
        emitStageCloud(denseDisplay, "dense");
        // 保存密集点云到项目目录
        saveStageCloud(denseCloud, resolveProjectFilePath("dense_cloud.ply"), "密集重建");
    }

    // =====================================================================
    // 组装输出（坐标系转换：OpenCV → 标准y-up）
    // =====================================================================
    // 密集重建全程在OpenCV坐标系(右手系,y-down,z-forward)中工作。
    // 导出时绕X轴旋转180°（同时翻转Y和Z），这是正规旋转(det=+1)，
    // 保持右手系，不会产生镜像/南北颠倒。
    PointCloudBuilder::instance()->convertToPCL(m_optPoints, output.sparseCloud);

    // 稀疏点云: y取反, z取反
    for (auto &pt : output.sparseCloud->points) {
        pt.y = -pt.y;
        pt.z = -pt.z;
    }

    // 密集点云: 创建副本并 y/z 取反
    if (denseCloud && !denseCloud->empty()) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr denseOutput(new pcl::PointCloud<pcl::PointXYZRGB>);
        denseOutput->resize(denseCloud->size());
        for (size_t i = 0; i < denseCloud->size(); ++i) {
            pcl::PointXYZRGB pt = denseCloud->points[i];
            pt.y = -pt.y;
            pt.z = -pt.z;
            denseOutput->points[i] = pt;
        }
        output.denseCloud = denseOutput;
        log(QString("[输出] 稀疏点云: %1 点, 密集点云: %2 点")
                .arg(output.sparseCloud ? static_cast<int>(output.sparseCloud->size()) : 0)
                .arg(static_cast<int>(output.denseCloud->size())));
    } else {
        log(QString("[输出] 稀疏点云: %1 点, 密集点云: 空（密集重建未生成有效点）")
                .arg(output.sparseCloud ? static_cast<int>(output.sparseCloud->size()) : 0));
    }

    output.cameraPoses = m_cameraPoses;
    output.totalPoints = static_cast<int>(m_optPoints.size());

    // ── 保存相机参数到 cameramatrix.txt ──
    if (!m_config.imageFiles.isEmpty()) {
        QString saveDir = !m_config.projectPath.isEmpty() ? m_config.projectPath
                                                          : QFileInfo(m_config.imageFiles[0]).absolutePath();
        QString savePath = saveDir + "/cameramatrix.txt";
        QFile file(savePath);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << "# Camera parameters estimated by SFMReconstructor\n";
            out << "# Format per camera: ImageName, then K(3x3), R(3x3), t(3x1)\n\n";
            for (const auto &pose : m_cameraPoses) {
                if (!pose.registered) continue;
                QString imgName = (pose.imageId >= 0 &&
                                  pose.imageId < m_config.imageFiles.size())
                                     ? QFileInfo(m_config.imageFiles[pose.imageId]).fileName()
                                     : QString("image_%1").arg(pose.imageId);
                out << imgName << "\n";

                // 内参矩阵 K
                cv::Mat K = pose.K.empty() ? cv::Mat::eye(3, 3, CV_64F) : pose.K;
                out << "# K (intrinsics)\n";
                for (int r = 0; r < 3; ++r) {
                    out << "K " << K.at<double>(r, 0) << " "
                        << K.at<double>(r, 1) << " "
                        << K.at<double>(r, 2) << "\n";
                }

                // 旋转矩阵 R
                cv::Mat R = pose.R.empty() ? cv::Mat::eye(3, 3, CV_64F) : pose.R;
                out << "# R (rotation)\n";
                for (int r = 0; r < 3; ++r) {
                    out << "R " << R.at<double>(r, 0) << " "
                        << R.at<double>(r, 1) << " "
                        << R.at<double>(r, 2) << "\n";
                }

                // 平移向量 t
                cv::Mat t = pose.t.empty() ? cv::Mat::zeros(3, 1, CV_64F) : pose.t;
                out << "# t (translation)\n";
                out << "t " << t.at<double>(0) << " "
                    << t.at<double>(1) << " "
                    << t.at<double>(2) << "\n";

                out << "\n";
            }
            file.close();
            log(QString("[相机参数] 已保存到 %1").arg(savePath));
        }
    }

    log(QString("[输出] 稀疏点云: %1 点").arg(m_optPoints.size()));
}

bool ReconstructionPipeline::denseReconstructionAvailable() const
{
    return m_sparseReady && !m_config.imageFiles.isEmpty();
}

bool ReconstructionPipeline::executeSparseReconstruction()
{
    // 当前由 execute() 统一执行并保存状态；
    // 单独执行入口保留给未来扩展。
    return m_sparseReady;
}

bool ReconstructionPipeline::executeDenseReconstruction()
{
    if (!denseReconstructionAvailable()) {
        log("[密集重建] 错误：没有可用的稀疏重建结果");
        return false;
    }

    m_stopRequested = false;
    QElapsedTimer totalTimer;
    totalTimer.start();

    log("========== 单独执行密集重建 ==========");

    ReconstructionOutput output;
    runDenseStages(output);

    if (m_stopRequested) {
        log("[密集重建] 收到停止请求");
        return false;
    }

    output.totalElapsedMs = totalTimer.elapsed();

    log(QString("[输出] 稀疏点云: %1 点").arg(m_optPoints.size()));

    setProgress("完成", 100);
    log(QString("========== 密集重建完成 (总耗时 %1 s) ==========")
        .arg(totalTimer.elapsed() / 1000.0, 0, 'f', 1));

    m_lastOutput = output;
    return true;
}

bool ReconstructionPipeline::isStopRequested() const
{
    return m_stopRequested;
}

void ReconstructionPipeline::exportPointCloud(const QString &filePath,
                                              const ReconstructionOutput &output)
{
    // 委托给统一的 saveStageCloud 实现（仅支持 PLY 格式）
    saveStageCloud(output.sparseCloud, filePath, "导出");
}

void ReconstructionPipeline::exportPointCloud(const QString &filePath)
{
    exportPointCloud(filePath, m_lastOutput);
}

void ReconstructionPipeline::saveProject(const QString &filePath,
                                         const ReconstructionOutput &output)
{
    (void)output;

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        log(QString("无法保存项目: %1").arg(filePath));
        return;
    }

    QJsonObject root;
    QJsonArray imageArray;
    for (const auto &img : m_config.imageFiles) {
        imageArray.append(img);
    }
    root["images"] = imageArray;
    root["totalPoints"] = output.totalPoints;

    QJsonDocument doc(root);
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();

    log(QString("项目已保存至: %1").arg(filePath));
}

void ReconstructionPipeline::saveProject(const QString &filePath)
{
    saveProject(filePath, m_lastOutput);
}

void ReconstructionPipeline::log(const QString &message)
{
    if (m_callbacks) {
        m_callbacks->onLog(message);
    }
}

void ReconstructionPipeline::setProgress(const QString &stage, int percent)
{
    if (m_callbacks) {
        m_callbacks->onProgress(stage, percent);
    }
}

void ReconstructionPipeline::emitStageCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                            const QString &stageName)
{
    if (!cloud || cloud->empty() || !m_callbacks) return;
    QVector<float> pts, cls;
    size_t count = cloud->size();
    size_t step = (count > 2000000) ? count / 2000000 : 1;
    pts.reserve(static_cast<int>((count / step) * 3));
    cls.reserve(static_cast<int>((count / step) * 3));
    for (size_t i = 0; i < count; i += step) {
        const auto &pt = cloud->points[i];
        pts << pt.x << pt.y << pt.z;
        cls << pt.r / 255.0f << pt.g / 255.0f << pt.b / 255.0f;
    }
    m_callbacks->onStagePointCloudReady(pts, cls, stageName);
}

void ReconstructionPipeline::saveStageCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                            const QString &filePath,
                                            const QString &stageLabel)
{
    if (!cloud || cloud->empty()) {
        log(QString("[%1] 点云为空，跳过保存").arg(stageLabel));
        return;
    }

    if (filePath.isEmpty()) {
        log(QString("[%1] 未配置输出路径，跳过保存").arg(stageLabel));
        return;
    }

    try {
        // 使用 PLY 二进制格式保存，避免 PCD 的 'rgb'/'rgba' 字段名不匹配问题
        pcl::io::savePLYFileBinary(filePath.toStdString(), *cloud);
        log(QString("[%1] 点云已保存: %2 (点数: %3)")
                .arg(stageLabel).arg(filePath).arg(cloud->size()));
    } catch (const std::exception &e) {
        log(QString("[%1] 点云保存异常: %2").arg(stageLabel).arg(e.what()));
    } catch (...) {
        log(QString("[%1] 点云保存异常: 未知错误").arg(stageLabel));
    }
}

QString ReconstructionPipeline::resolveProjectFilePath(const QString &fileName) const
{
    QString saveDir = !m_config.projectPath.isEmpty()
                          ? m_config.projectPath
                          : (!m_config.imageFiles.isEmpty()
                                 ? QFileInfo(m_config.imageFiles[0]).absolutePath()
                                 : QString());
    if (saveDir.isEmpty()) return QString();
    return saveDir + "/" + fileName;
}