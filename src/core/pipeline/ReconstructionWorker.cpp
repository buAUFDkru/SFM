#include "ReconstructionWorker.h"

#include <QDir>

ReconstructionWorker::ReconstructionWorker(ReconstructionPipeline *pipeline, QObject *parent)
    : QObject(parent)
    , m_pipeline(pipeline)
    , m_lastFlushTime(std::chrono::steady_clock::now())
{
    if (m_pipeline) {
        m_pipeline->setCallbacks(this);
        m_pipeline->setProgressSink(this);
    }
}

ReconstructionWorker::~ReconstructionWorker()
{
    if (m_pipeline && m_pipeline->progressSink() == this)
        m_pipeline->setProgressSink(nullptr);
    flush();
}

void ReconstructionWorker::runFullPipeline()
{
    if (!m_pipeline) {
        emit finished(false);
        return;
    }

    m_running = true;
    m_pipeline->execute();
    m_running = false;

    autoSaveIfNeeded();
    flush();

    emit finished(m_pipeline->hasCloud(false));
}

void ReconstructionWorker::runDenseReconstruction()
{
    if (!m_pipeline) {
        emit finished(false);
        return;
    }

    m_running = true;
    bool ok = m_pipeline->executeDenseReconstruction();
    m_running = false;

    autoSaveIfNeeded();
    flush();

    emit finished(ok);
}

void ReconstructionWorker::setDenseReconstructionMethod(const QString &method,
                                                        const QMap<QString, QVariant> &params)
{
    if (m_pipeline) {
        m_pipeline->setDenseReconstructionMethod(method, params);
    }
}

void ReconstructionWorker::requestStop()
{
    if (m_pipeline) {
        m_pipeline->requestStop();
    }
}

void ReconstructionWorker::setAutoSaveDirectory(const QString &dir)
{
    m_autoSaveDir = dir;
}

void ReconstructionWorker::autoSaveIfNeeded()
{
    if (!m_pipeline || m_autoSaveDir.isEmpty())
        return;

    QDir dir;
    dir.mkpath(m_autoSaveDir);
}

void ReconstructionWorker::onProgress(const QString &stage, int percent)
{
    // 无 ProgressSink 回退路径（理论上 Worker 已注入 sink，仍保持回调兼容）
    onStage(stage, percent);
}

void ReconstructionWorker::onStage(const QString &stage, int percent)
{
    maybeFlushLog();
    emit progressChanged(stage, percent);
}

void ReconstructionWorker::onLog(const QString &message)
{
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        m_logBuffer.append(message);
    }
    maybeFlushLog();
}

void ReconstructionWorker::flush()
{
    maybeFlushLog(true);
}

void ReconstructionWorker::onStagePointCloudReady(const QVector<float> &points,
                                                  const QVector<float> &colors,
                                                  const QString &stageName)
{
    emit stagePointCloudReady(points, colors, stageName);
}

void ReconstructionWorker::onDenseReconstructionMethodRequested()
{
    flush();
    emit denseReconstructionMethodRequested();
}

void ReconstructionWorker::maybeFlushLog(bool force)
{
    QStringList batch;
    {
        std::lock_guard<std::mutex> lock(m_logMutex);

        if (m_logBuffer.isEmpty())
            return;

        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - m_lastFlushTime)
                             .count();

        bool shouldFlush = force || m_logBuffer.size() >= flushBatchSize ||
                           elapsedMs >= flushIntervalMs;
        if (!shouldFlush)
            return;

        batch = std::move(m_logBuffer);
        m_logBuffer.clear();
        m_lastFlushTime = now;
    }

    emit logBatchEmitted(batch);
}
