#ifndef RECONSTRUCTIONWORKER_H
#define RECONSTRUCTIONWORKER_H

#include <QMap>
#include <QObject>
#include <QStringList>
#include <QVariant>
#include <QVector>
#include <chrono>
#include <mutex>

#include "algorithm/ProgressSink.h"
#include "ReconstructionPipeline.h"

class ReconstructionPipeline;

/**
 * @brief 重建工作线程类
 *
 * 负责在工作线程中执行 ReconstructionPipeline 的各个阶段，
 * 并通过 Qt 信号将进度/日志/点云等事件发回主线程 UI。
 *
 * 同时实现 ProgressSink（ADR-003）：进度/日志优先经 Pipeline::setProgressSink 注入；
 * 日志批量缓冲，在 batchSize / onStage / flush() 时刷出，不再使用监视线程。
 */
class ReconstructionWorker : public QObject,
                             public IReconstructionCallbacks,
                             public sfm::ProgressSink
{
    Q_OBJECT

public:
    explicit ReconstructionWorker(ReconstructionPipeline *pipeline, QObject *parent = nullptr);
    ~ReconstructionWorker() override;

    /// 执行完整流水线
    void runFullPipeline();

    /// 单独执行密集重建（要求 pipeline 已有稀疏重建结果）
    void runDenseReconstruction();

    /// 设置用户选择的密集重建方法（由 UI 调用）
    void setDenseReconstructionMethod(const QString &method,
                                      const QMap<QString, QVariant> &params);

    /// 请求停止当前重建
    void requestStop();

    /// 是否正在运行
    bool isRunning() const { return m_running; }

    /// 设置自动保存目录；非空时完成重建后自动保存点云
    void setAutoSaveDirectory(const QString &dir);

    // --- IReconstructionCallbacks ---
    void onProgress(const QString &stage, int percent) override;
    void onLog(const QString &message) override;
    void onStagePointCloudReady(const QVector<float> &points,
                                const QVector<float> &colors,
                                const QString &stageName) override;
    void onDenseReconstructionMethodRequested() override;

    // --- ProgressSink ---
    void onStage(const QString &stage, int percent) override;
    void flush() override;

signals:
    void progressChanged(const QString &stage, int percent);
    void logEmitted(const QString &message);
    /// 批量日志信号，一次发射多条日志，减少 UI 事件队列压力
    void logBatchEmitted(const QStringList &messages);
    void stagePointCloudReady(const QVector<float> &points,
                              const QVector<float> &colors,
                              const QString &stageName);
    void denseReconstructionMethodRequested();
    void finished(bool success);

private:
    void maybeFlushLog(bool force = false);
    void autoSaveIfNeeded();

    ReconstructionPipeline *m_pipeline;
    bool m_running = false;
    QString m_autoSaveDir;

    std::mutex m_logMutex;
    QStringList m_logBuffer;
    std::chrono::steady_clock::time_point m_lastFlushTime;
    const int flushIntervalMs = 200;
    const int flushBatchSize = 10;
};

#endif // RECONSTRUCTIONWORKER_H
