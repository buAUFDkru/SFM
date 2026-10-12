#include "ReconstructionPipeline.h"
#include "algorithm/SfmLogger.h"
#include "algorithm/ThreadPool.h"
#include "algorithm/Triangulate.h"
#include "data/ReconstructionOutput.h"
#include "pipeline/DenseReconstructionFactory.h"
#include "pipeline/DenseReconstructionStage.h"
#include "pipeline/FeatureExtractionStage.h"
#include "pipeline/ProjectRepository.h"
#include "pipeline/ReconstructionConfigIO.h"
#include "pipeline/SparseMappingStage.h"
#include "viewmodel/SceneViewMapper.h"
#include <QElapsedTimer>
#include <QFileInfo>
#include <opencv2/core.hpp>

ReconstructionPipeline::ReconstructionPipeline()
    : m_config(sfm::makeDefaultReconstructionConfig())
    , m_lastOutput(std::make_unique<ReconstructionOutput>())
    , m_triangulate(std::make_unique<Triangulate>())
{
    DenseReconstructionFactory::instance().registerBuiltinMethods();
}
Triangulate &ReconstructionPipeline::triangulator() { return *m_triangulate; }
const Triangulate &ReconstructionPipeline::triangulator() const { return *m_triangulate; }
ReconstructionPipeline::~ReconstructionPipeline()
{
    if (m_sfmLoggerInstalled) {
        SfmLogger::instance().uninstallAsMessageHandler();
        m_sfmLoggerInstalled = false;
    }
    if (SfmLogger::instance().messageHandlerInstallCount() == 0)
        SfmLogger::instance().clearCallbacks();
}
void ReconstructionPipeline::setCallbacks(IReconstructionCallbacks *callbacks)
{
    m_callbacks = callbacks;
    if (m_callbacks) {
        SfmLogger::instance().setDefaultCallback([this](const QString &msg) { log(msg); });
        if (!m_sfmLoggerInstalled) {
            SfmLogger::instance().installAsMessageHandler();
            m_sfmLoggerInstalled = true;
        }
    } else if (m_sfmLoggerInstalled) {
        SfmLogger::instance().uninstallAsMessageHandler();
        m_sfmLoggerInstalled = false;
        if (SfmLogger::instance().messageHandlerInstallCount() == 0)
            SfmLogger::instance().clearCallbacks();
    }
}
void ReconstructionPipeline::setProgressSink(sfm::ProgressSink *sink)
{
    m_progress = sink;
}
void ReconstructionPipeline::waitForDenseReconstructionMethod()
{
    std::unique_lock<std::mutex> lock(m_denseMethodMutex);
    m_denseMethodCv.wait(lock, [this] {
        return m_denseMethodReady || m_cancel.isStopRequested();
    });
}
void ReconstructionPipeline::setDenseReconstructionMethod(
    const QString &method, const QMap<QString, QVariant> &params)
{
    {
        std::lock_guard<std::mutex> lock(m_denseMethodMutex);
        m_config.denseReconstructionMethod = method;
        m_config.denseReconstructionParams = params;
        m_denseMethodReady = true;
    }
    m_denseMethodCv.notify_all();
}
void ReconstructionPipeline::requestStop()
{
    m_cancel.requestStop();
    m_denseMethodCv.notify_all();
}
void ReconstructionPipeline::setConfig(const ReconstructionConfig &config) { m_config = config; }
ReconstructionConfig ReconstructionPipeline::config() const { return m_config; }
void ReconstructionPipeline::execute()
{
    m_cancel.reset();
    m_denseMethodReady = false;
    ReconstructionOutput output;
    if (m_config.imageFiles.isEmpty()) {
        log(QStringLiteral("错误: 未加载图像"));
        return;
    }
    log(QStringLiteral("========== SFM 重建流水线启动 =========="));
    log(QStringLiteral("[配置] 图像数量: %1").arg(m_config.imageFiles.size()));
    m_config.imageFiles.sort();
    cv::setNumThreads(1);
    sfm::ThreadPool &pool =
        sfm::globalThreadPool(static_cast<std::size_t>(m_config.threadCount));
    log(QStringLiteral("[线程池] 全局共享线程池, 工作线程数: %1").arg(pool.thread_count()));
    QElapsedTimer totalTimer;
    totalTimer.start();
    const auto logFn = [this](const QString &m) { log(m); };
    const auto progressFn = [this](const QString &s, int p) { setProgress(s, p); };
    sfm::FeatureExtractionInput featIn{m_config.imageFiles, m_config.descriptorType,
                                       m_config.descriptorParams};
    if (!sfm::FeatureExtractionStage::run(featIn, &m_features, &pool, &m_cancel, logFn,
                                          progressFn))
        return;
    const sfm::SparseMappingResult sparse = sfm::SparseMappingStage::run(
        sfm::SparseMappingStage::fromConfig(
            m_config, m_features,
            resolveProjectFilePath(QStringLiteral("clouds/sparse.ply"))),
        *m_triangulate, &pool, &m_cancel, m_callbacks, logFn, progressFn);
    if (sparse.cancelled || !sparse.ok)
        return;
    m_features = sparse.features;
    m_matchPairs = sparse.matchPairs;
    m_cameraPoses = sparse.cameraPoses;
    m_optPoints = sparse.optPoints;
    m_K = sparse.K.clone();
    m_sparseReady = true;
    if (sparse.sparseDisplay) {
        m_lastOutput->sparseCloud = sparse.sparseDisplay;
        m_lastOutput->cameraPoses = m_cameraPoses;
        m_lastOutput->totalPoints = static_cast<int>(m_optPoints.size());
    }
    runDenseStages(output);
    if (m_cancel.isStopRequested()) {
        log(QStringLiteral("[流水线] 收到停止请求"));
        return;
    }
    output.totalElapsedMs = totalTimer.elapsed();
    setProgress(QStringLiteral("完成"), 100);
    log(QStringLiteral("========== 重建流水线完成 (总耗时 %1 s) ==========")
            .arg(totalTimer.elapsed() / 1000.0, 0, 'f', 1));
    *m_lastOutput = output;
}
void ReconstructionPipeline::runDenseStages(ReconstructionOutput &output)
{
    if (!m_sparseReady) {
        log(QStringLiteral("[密集重建] 错误：未执行稀疏重建"));
        return;
    }
    const QString camPath =
        m_config.imageFiles.isEmpty()
            ? QString()
            : ((!m_config.projectPath.isEmpty()
                    ? m_config.projectPath
                    : QFileInfo(m_config.imageFiles[0]).absolutePath())
               + QStringLiteral("/cameramatrix.txt"));
    sfm::DenseStageHooks hooks;
    hooks.requestMethodIfEmpty = [this]() {
        if (m_callbacks) m_callbacks->onDenseReconstructionMethodRequested();
        waitForDenseReconstructionMethod();
    };
    hooks.methodName = [this]() { return m_config.denseReconstructionMethod; };
    hooks.methodParams = [this]() { return m_config.denseReconstructionParams; };
    sfm::DenseReconstructionStage::run(
        sfm::DenseReconstructionStage::fromState(
            m_config, resolveProjectFilePath(QStringLiteral("clouds/dense.ply")), camPath,
            m_cameraPoses, m_matchPairs, m_features, m_K, m_optPoints,
            m_lastOutput->sparseCloud),
        &output, &m_cancel, m_callbacks, hooks,
        [this](const QString &m) { log(m); },
        [this](const QString &s, int p) { setProgress(s, p); });
}
bool ReconstructionPipeline::denseReconstructionAvailable() const
{
    return m_sparseReady && !m_config.imageFiles.isEmpty();
}
bool ReconstructionPipeline::executeSparseReconstruction() { return m_sparseReady; }
bool ReconstructionPipeline::executeDenseReconstruction()
{
    if (!denseReconstructionAvailable()) {
        log(QStringLiteral("[密集重建] 错误：没有可用的稀疏重建结果"));
        return false;
    }
    m_cancel.reset();
    m_denseMethodReady = false;
    QElapsedTimer t;
    t.start();
    log(QStringLiteral("========== 单独执行密集重建 =========="));
    ReconstructionOutput output;
    runDenseStages(output);
    if (m_cancel.isStopRequested()) {
        log(QStringLiteral("[密集重建] 收到停止请求"));
        return false;
    }
    output.totalElapsedMs = t.elapsed();
    setProgress(QStringLiteral("完成"), 100);
    log(QStringLiteral("========== 密集重建完成 (总耗时 %1 s) ==========")
            .arg(t.elapsed() / 1000.0, 0, 'f', 1));
    *m_lastOutput = output;
    return true;
}
bool ReconstructionPipeline::isStopRequested() const { return m_cancel.isStopRequested(); }
void ReconstructionPipeline::exportPointCloud(const QString &filePath,
                                              const ReconstructionOutput &output)
{
    sfm::ProjectRepository::saveCloudPly(
        output.sparseCloud, filePath, QStringLiteral("导出"),
        [this](const QString &msg) { log(msg); });
}
void ReconstructionPipeline::exportPointCloud(const QString &filePath)
{
    exportPointCloud(filePath, *m_lastOutput);
}
bool ReconstructionPipeline::saveProject(QString *error)
{
    return sfm::ProjectRepository::saveProject(
        m_config, &m_manifest, m_cameraPoses, m_matchPairs, *m_lastOutput,
        [this](const QString &msg) { log(msg); }, error);
}
bool ReconstructionPipeline::loadProject(const QString &path, QString *error)
{
    m_cancel.reset();
    m_features.clear();
    m_optPoints.clear();
    return sfm::ProjectRepository::loadProject(
        path, &m_config, &m_manifest, &m_cameraPoses, &m_matchPairs, &m_K, &m_sparseReady,
        m_lastOutput.get(), [this](const QString &msg) { log(msg); }, error);
}
void ReconstructionPipeline::log(const QString &message)
{
    if (m_progress) {
        m_progress->onLog(message);
        return;
    }
    if (m_callbacks)
        m_callbacks->onLog(message);
}
void ReconstructionPipeline::setProgress(const QString &stage, int percent)
{
    if (m_progress) {
        m_progress->onStage(stage, percent);
        return;
    }
    if (m_callbacks)
        m_callbacks->onProgress(stage, percent);
}
const ReconstructionOutput &ReconstructionPipeline::lastOutput() const
{
    return *m_lastOutput;
}
sfm::view::ReconstructionView ReconstructionPipeline::lastView() const
{
    return sfm::view::toReconstructionView(*m_lastOutput);
}
bool ReconstructionPipeline::hasCloud(bool dense) const
{
    const auto &c = dense ? m_lastOutput->denseCloud : m_lastOutput->sparseCloud;
    return c && !c->empty();
}
QString ReconstructionPipeline::resolveProjectFilePath(const QString &fileName) const
{
    return sfm::ProjectRepository::resolveOutputPath(m_config, fileName);
}
