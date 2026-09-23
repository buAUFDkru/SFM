#ifndef RECONSTRUCTIONWORKER_H
#define RECONSTRUCTIONWORKER_H

#include <QMap>
#include <QObject>
#include <QStringList>
#include <QVariant>
#include <QVector>
#include <chrono>
#include <mutex>

#include "ReconstructionPipeline.h"

class ReconstructionPipeline;

/**
 * @brief 重建工作线程类
 *
 * 负责在工作线程中执行 ReconstructionPipeline 的各个阶段，
 * 并通过 Qt 信号将进度/日志/点云等事件发回主线程 UI。
 *
 * ReconstructionPipeline 本身不继承 QObject，此 Worker 是它唯一的"线程外壳"。
 *
 * 日志批量刷新机制：
 * 线程池工作线程产生的日志通过 onLog() 进入缓冲区 m_logBuffer。
 * 当缓冲区达到一定数量或距上次刷新超过一定时间时，
 * 直接 emit logBatchEmitted（通过 Qt::QueuedConnection 投递到 UI 线程）。
 * 不依赖 QTimer，因为 Worker 线程在 execute() 期间不运行事件循环，
 * QTimer 的 timeout 事件无法被处理。
 */
class ReconstructionWorker : public QObject, public IReconstructionCallbacks
{
    Q_OBJECT

public:
    explicit ReconstructionWorker(ReconstructionPipeline *pipeline, QObject *parent = nullptr);

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

    // --- IReconstructionCallbacks 实现 ---
    void onProgress(const QString &stage, int percent) override;
    void onLog(const QString &message) override;
    void onStagePointCloudReady(const QVector<float> &points,
                                const QVector<float> &colors,
                                const QString &stageName) override;
    void onDenseReconstructionMethodRequested() override;

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
    /// 检查是否需要刷新缓冲区（按数量或时间阈值），需要则发射 logBatchEmitted
    void maybeFlushLog(bool force = false);

    void autoSaveIfNeeded();

    ReconstructionPipeline *m_pipeline;
    bool m_running = false;
    QString m_autoSaveDir;

    // 日志批量刷新：缓冲区 + 时间/数量阈值
    // 不使用 QTimer，因为 Worker 线程在 execute() 期间不运行事件循环。
    // 通过时间戳判断：距上次刷新超过 flushIntervalMs 或缓冲区达到 flushBatchSize 时触发。
    std::mutex m_logMutex;
    QStringList m_logBuffer;
    std::chrono::steady_clock::time_point m_lastFlushTime;
    const int flushIntervalMs = 200;   // 最小刷新间隔
    const int flushBatchSize = 10;     // 缓冲区达到此数量时强制刷新
};

#endif // RECONSTRUCTIONWORKER_H
