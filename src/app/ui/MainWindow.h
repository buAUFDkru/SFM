#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>

#include "pipeline/ReconstructionPipeline.h"
#include "pipeline/PipelineRunner.h"

class ConfigPanel;
class LogConsole;
class PointCloudWidget;
class PluginRegistry;
class QLabel;
class QSplitter;

/**
 * @brief SFM Reconstructor 主窗口
 *
 * 布局：左侧 ConfigPanel（schema 驱动参数） | 中间 PointCloudWidget | 右侧 LogConsole。
 * 重建在 PipelineRunner 工作线程中执行，pipeline 信号 → UI 槽（跨线程 QueuedConnection）。
 *
 * pluginsDir：插件扫描目录（默认 applicationDirPath()/plugins）。
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(const QString &pluginsDir, QWidget *parent = nullptr);
    ~MainWindow() override;

    /// 返回内部 PluginRegistry（测试自检用）
    PluginRegistry *registry() const;

    /// 装配当前 UI 中的图像文件列表（测试自检用）
    QStringList currentImageFiles() const;
    int currentImageCount() const;
    void setImageFilesForTest(const QStringList &files);
    void setProjectPathForTest(const QString &path);

    /// 同步执行一次完整重建并将结果存入 lastOutput（测试自检用，阻塞直至完成）
    void runPipelineSyncForTest();

    /// 最近一次重建输出（测试自检用）
    const ReconstructionOutput &lastOutput() const { return m_pipeline ? m_pipeline->lastOutput() : m_lastFallback; }

private slots:
    void onStartReconstruction();
    void onStopReconstruction();
    void onDenseReconstruction();
    void onExportPointCloud();
    void onSaveProjectFile();
    void onResetView();
    void onTopView();
    void onFrontView();
    void onSideView();
    void onWireframeMode(bool enabled);
    void onImportImages();
    void onImportDirectory();
    void onPipelineProgress(const QString &stage, int percent);
    void onPipelineLog(const QString &message);
    void onPipelineLogBatch(const QStringList &messages);
    void onPipelineFinished(bool success);
    void onStagePointCloudReady(const QVector<float> &points,
                                const QVector<float> &colors,
                                const QString &stageName);
    void onCameraPosesReady(const QVector<float> &poses);
    void updateStatusLabel();

private:
    void setupMenuBar();
    void setupStatusBar();
    void setupCentralWidget();
    void connectSignals();
    void loadAndDisplayCloudFromOutput();

    PluginRegistry *m_registry = nullptr;
    QSplitter *m_splitter = nullptr;
    ConfigPanel *m_configPanel = nullptr;
    PointCloudWidget *m_pointCloudWidget = nullptr;
    LogConsole *m_logConsole = nullptr;
    QLabel *m_statusLabel = nullptr;
    QLabel *m_threadLabel = nullptr;

    ReconstructionPipeline *m_pipeline = nullptr;
    PipelineRunner *m_pipelineRunner = nullptr;
    ReconstructionOutput m_lastFallback;  ///< 无 pipeline 时的兜底输出
};

#endif // MAINWINDOW_H