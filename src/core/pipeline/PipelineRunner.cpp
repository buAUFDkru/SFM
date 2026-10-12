#include "PipelineRunner.h"
#include "ReconstructionPipeline.h"
#include "ReconstructionWorker.h"
#include "algorithm/Triangulate.h"

#include <QDebug>

PipelineRunner::PipelineRunner(ReconstructionPipeline *pipeline, QObject *parent)
    : QObject(parent)
    , m_pipeline(pipeline)
    , m_worker(new ReconstructionWorker(pipeline, nullptr))
{
    // 将 Worker 移入工作线程；Worker 不能有 parent，否则 moveToThread 会失败
    m_worker->moveToThread(&m_workerThread);

    // 注意：不在构造时调用 setPriority，因为线程尚未启动会触发警告
    // "QThread::setPriority: Cannot set priority, thread is not running"。
    // 改为在 startWithMode 中通过 start(QThread::LowPriority) 启动线程。

    // 转发 Worker 的信号到 PipelineRunner（跨线程自动队列连接）
    connect(m_worker, &ReconstructionWorker::progressChanged,
            this, &PipelineRunner::progressChanged);
    connect(m_worker, &ReconstructionWorker::logEmitted,
            this, &PipelineRunner::logEmitted);
    connect(m_worker, &ReconstructionWorker::logBatchEmitted,
            this, &PipelineRunner::logBatchEmitted);
    connect(m_worker, &ReconstructionWorker::stagePointCloudReady,
            this, &PipelineRunner::stagePointCloudReady);
    connect(m_worker, &ReconstructionWorker::denseReconstructionMethodRequested,
            this, &PipelineRunner::denseReconstructionMethodRequested);
    connect(m_worker, &ReconstructionWorker::finished,
            this, &PipelineRunner::onWorkerFinished);

    // 转发本流水线 Triangulate 的增量更新（跨线程自动队列连接）
    if (m_pipeline) {
        connect(&m_pipeline->triangulator(), &Triangulate::incrementalUpdate,
                this, &PipelineRunner::incrementalUpdate);
    }
}

PipelineRunner::~PipelineRunner()
{
    stop();
    if (m_workerThread.isRunning()) {
        if (!m_workerThread.wait(5000)) {
            qWarning().noquote() << "[PipelineRunner] 工作线程未在5秒内停止，强制终止";
            m_workerThread.terminate();
            m_workerThread.wait();
        }
    }
    // 通过 deleteLater 在 Worker 所在线程安全销毁，避免跨线程销毁 QObject 警告。
    // 若线程已退出（事件循环停止），deleteLater 会在当前线程（主线程）立即处理，
    // 此时 Worker 已无定时器等需要线程亲和性的对象，可安全销毁。
    if (m_worker) {
        m_worker->deleteLater();
        m_worker = nullptr;
    }
}

void PipelineRunner::start()
{
    startWithMode(RunMode::Full);
}

void PipelineRunner::setAutoSaveDirectory(const QString &dir)
{
    if (m_worker) {
        m_worker->setAutoSaveDirectory(dir);
    }
}

void PipelineRunner::startDenseReconstruction()
{
    if (!m_pipeline || !m_pipeline->denseReconstructionAvailable()) {
        qWarning().noquote() << "[PipelineRunner] 没有可用的稀疏重建结果，无法执行密集重建";
        return;
    }
    startWithMode(RunMode::DenseOnly);
}

void PipelineRunner::startWithMode(RunMode mode)
{
    if (isRunning()) return;

    m_runMode = mode;

    if (m_workerThread.isRunning()) {
        // 工作线程已在运行，直接排队调度任务
        if (mode == RunMode::DenseOnly) {
            QMetaObject::invokeMethod(m_worker, &ReconstructionWorker::runDenseReconstruction,
                                      Qt::QueuedConnection);
        } else {
            QMetaObject::invokeMethod(m_worker, &ReconstructionWorker::runFullPipeline,
                                      Qt::QueuedConnection);
        }
    } else {
        // 线程未运行，连接 started 信号，启动后自动执行对应任务。
        // 必须先断开之前所有的 started 连接！
        // Qt::UniqueConnection 只防止"相同信号-槽对"重复，但 runFullPipeline / runDenseReconstruction
        // 是不同的槽，会同时连接到 started，导致线程启动时
        // 多个槽都被调用，造成重复执行。
        disconnect(&m_workerThread, &QThread::started, nullptr, nullptr);

        if (mode == RunMode::DenseOnly) {
            connect(&m_workerThread, &QThread::started,
                    m_worker, &ReconstructionWorker::runDenseReconstruction);
        } else {
            connect(&m_workerThread, &QThread::started,
                    m_worker, &ReconstructionWorker::runFullPipeline);
        }
        // 通过 start(priority) 重载设置线程优先级，避免线程未启动时调用 setPriority 的警告。
        // LowPriority 让 UI 主线程优先调度，避免 CPU 密集型重建导致 UI 卡顿。
        m_workerThread.start(QThread::LowPriority);
    }
}

void PipelineRunner::stop()
{
    if (m_worker) {
        m_worker->requestStop();
    }
    m_workerThread.quit();
}

bool PipelineRunner::isRunning() const
{
    return m_worker && m_worker->isRunning();
}

void PipelineRunner::waitForFinished()
{
    if (m_workerThread.isRunning()) {
        m_workerThread.quit();
        if (!m_workerThread.wait(5000)) {
            qWarning().noquote() << "[PipelineRunner] 工作线程未在5秒内停止，强制终止";
            m_workerThread.terminate();
            m_workerThread.wait();
        }
    }
}

void PipelineRunner::setDenseReconstructionMethod(const QString &method,
                                                  const QMap<QString, QVariant> &params)
{
    if (m_worker) {
        m_worker->setDenseReconstructionMethod(method, params);
    }
}

void PipelineRunner::onWorkerFinished(bool success)
{
    emit finished(success);
    m_workerThread.quit();
}
