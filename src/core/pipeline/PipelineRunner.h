#ifndef PIPELINERUNNER_H
#define PIPELINERUNNER_H

#include <QMap>
#include <QObject>
#include <QThread>
#include <QVariant>
#include <QVector>
#include "data/SfmTypes.h"

class ReconstructionPipeline;
class ReconstructionWorker;

/**
 * @brief 流水线运行器类
 *
 * 负责管理工作线程和 ReconstructionWorker 的生命周期，
 * 将 Worker 的信号转发到主线程（UI线程）。
 */
class PipelineRunner : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param pipeline 要执行的重建流水线实例
     * @param parent 父对象
     */
    explicit PipelineRunner(ReconstructionPipeline *pipeline, QObject *parent = nullptr);

    /**
     * @brief 析构函数
     *
     * 确保工作线程已停止并等待其结束。
     */
    ~PipelineRunner() override;

    /**
     * @brief 启动完整重建流水线
     *
     * 在工作线程中执行完整重建，完成后发出 finished 信号。
     */
    void start();

    /**
     * @brief 设置自动保存目录，完成重建后由 Worker 在工作线程中自动保存
     */
    void setAutoSaveDirectory(const QString &dir);

    /**
     * @brief 单独启动密集重建阶段
     *
     * 要求流水线已先完成稀疏重建，保存有中间结果。
     * 在工作线程中执行密集重建阶段，完成后发出 finished 信号。
     */
    void startDenseReconstruction();

    /**
     * @brief 请求停止重建
     *
     * 设置停止标志，工作线程将在当前阶段完成后退出。
     */
    void stop();

    /**
     * @brief 检查是否正在运行
     * @return 是否正在执行重建
     */
    bool isRunning() const;

    /**
     * @brief 等待工作线程结束
     */
    void waitForFinished();

signals:
    /**
     * @brief 进度改变信号
     * @param stage 当前阶段名称
     * @param percent 进度百分比
     */
    void progressChanged(const QString &stage, int percent);

    /**
     * @brief 日志消息信号
     * @param message 日志文本
     */
    void logEmitted(const QString &message);

    /**
     * @brief 批量日志消息信号
     * @param messages 多条日志文本
     */
    void logBatchEmitted(const QStringList &messages);

    /**
     * @brief 重建完成信号
     * @param success 是否成功完成
     */
    void finished(bool success);

    /**
     * @brief 点云数据就绪信号
     * @param points 点坐标数组（x1,y1,z1,x2,y2,z2,...）
     * @param colors 颜色数组（r1,g1,b1,r2,g2,b2,...）
     */
    void pointCloudReady(const QVector<float> &points, const QVector<float> &colors);

    /**
     * @brief 增量重建实时更新信号
     * @param points 点坐标 (x1,y1,z1,x2,y2,z2,...)
     * @param colors 颜色 (r1,g1,b1,r2,g2,b2,...) 范围0-1
     * @param cameraData 相机位姿 (cx,cy,cz, dx,dy,dz 每个相机6个float)
     * @param reset 是否重置显示
     */
    void incrementalUpdate(const QVector<float> &points,
                           const QVector<float> &colors,
                           const QVector<float> &cameraData,
                           bool reset);

    /**
     * @brief 阶段性点云就绪信号
     * @param points 点坐标数组（x1,y1,z1,x2,y2,z2,...）
     * @param colors 颜色数组（r1,g1,b1,r2,g2,b2,...）范围0-1
     * @param stageName 阶段名称（"sparse"/"dense"）
     */
    void stagePointCloudReady(const QVector<float> &points,
                              const QVector<float> &colors,
                              const QString &stageName);

    /**
     * @brief 请求 UI 选择密集重建方法
     *
     * 工作线程已暂停，等待 UI 调用 setDenseReconstructionMethod()。
     */
    void denseReconstructionMethodRequested();

public slots:
    /**
     * @brief 设置用户选择的密集重建方法（转发给 Worker）
     */
    void setDenseReconstructionMethod(const QString &method,
                                      const QMap<QString, QVariant> &params);

private slots:
    void onWorkerFinished(bool success);

private:
    enum class RunMode { Full, DenseOnly };

    void startWithMode(RunMode mode);

    ReconstructionPipeline *m_pipeline;  ///< 重建流水线实例
    ReconstructionWorker *m_worker;      ///< 工作线程中的执行器
    QThread m_workerThread;              ///< 工作线程
    RunMode m_runMode = RunMode::Full;   ///< 当前运行模式
};

#endif // PIPELINERUNNER_H
