#include "ReconstructionWorker.h"

#include <QDir>

ReconstructionWorker::ReconstructionWorker(ReconstructionPipeline *pipeline, QObject *parent)
    : QObject(parent)
    , m_pipeline(pipeline)
    , m_lastFlushTime(std::chrono::steady_clock::now())
{
    if (m_pipeline) {
        m_pipeline->setCallbacks(this);
    }
    // 不再使用 QTimer：
    // Worker 线程在 execute() 期间不运行事件循环，QTimer 的 timeout 无法被处理。
    // 改为在 onLog() 中按时间/数量阈值触发 maybeFlushLog()。
}

void ReconstructionWorker::runFullPipeline()
{
    if (!m_pipeline) {
        emit finished(false);
        return;
    }

    m_running = true;
    ReconstructionOutput output = m_pipeline->execute();
    m_running = false;

    autoSaveIfNeeded();

    // 流水线结束后强制刷新剩余日志，确保所有日志显示
    maybeFlushLog(true);

    // 对于完整流水线，只要稀疏点云生成即算基本成功
    bool success = output.sparseCloud && !output.sparseCloud->empty();
    emit finished(success);
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

    // 流水线结束后强制刷新剩余日志
    maybeFlushLog(true);

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
    if (!m_pipeline || m_autoSaveDir.isEmpty()) return;

    QDir dir;
    dir.mkpath(m_autoSaveDir);

    // 注：稀疏/密集点云已由 ReconstructionPipeline::saveStageCloud
    //   在各阶段完成时自动保存到项目目录，此处不再重复保存点云。
}

void ReconstructionWorker::onProgress(const QString &stage, int percent)
{
    emit progressChanged(stage, percent);
}

void ReconstructionWorker::onLog(const QString &message)
{
    // 高频日志进入缓冲区，由 maybeFlushLog 按时间/数量阈值批量发射，
    // 避免淹没 UI 事件队列。
    // 此函数可能被线程池工作线程并发调用，需加锁保护缓冲区。
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        m_logBuffer.append(message);
    }
    maybeFlushLog();
}

void ReconstructionWorker::onStagePointCloudReady(const QVector<float> &points,
                                                  const QVector<float> &colors,
                                                  const QString &stageName)
{
    emit stagePointCloudReady(points, colors, stageName);
}

void ReconstructionWorker::onDenseReconstructionMethodRequested()
{
    // 切换 UI 前先刷新已缓冲的日志
    maybeFlushLog(true);
    emit denseReconstructionMethodRequested();
}

void ReconstructionWorker::maybeFlushLog(bool force)
{
    QStringList batch;
    {
        std::lock_guard<std::mutex> lock(m_logMutex);

        if (m_logBuffer.isEmpty()) return;

        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - m_lastFlushTime).count();

        // 触发条件：强制刷新 / 缓冲区达到批次大小 / 距上次刷新超过间隔
        bool shouldFlush = force ||
                           m_logBuffer.size() >= flushBatchSize ||
                           elapsedMs >= flushIntervalMs;
        if (!shouldFlush) return;

        batch = std::move(m_logBuffer);
        m_logBuffer.clear();
        m_lastFlushTime = now;
    }

    // 批量发射，UI 端一次性处理多条日志，减少信号投递和 UI 重绘次数。
    // emit 是线程安全的，Qt 会通过 QueuedConnection 把信号投递到 UI 线程事件队列。
    emit logBatchEmitted(batch);
}
