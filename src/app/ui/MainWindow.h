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
class QAction;
class QLabel;
class QMenu;
class QSplitter;

/// 相机金字塔几何数据（线框 + 底面四角 + 对应图像路径）
struct CameraFrustumGeometry
{
    QVector<float> lines;               ///< 线框成对端点（显示坐标系 y-up）
    QVector<QVector3D> quadCorners;     ///< 每相机 4 个底面角（缩略图贴图用）
    QStringList imagePaths;             ///< 每相机对应图像路径（未注册/越界为空串）
};

/**
 * @brief SFM Reconstructor 主窗口
 *
 * 布局：左侧 ConfigPanel（schema 驱动参数） | 中间 PointCloudWidget | 右侧 LogConsole。
 * 重建在 PipelineRunner 工作线程中执行，pipeline 信号 → UI 槽（跨线程 QueuedConnection）。
 *
 * 项目管理：项目状态（目录/名称/修改标记）由本窗口持有，菜单提供
 * 新建/打开/保存/最近打开；重建完成后自动保存项目结果。
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

    /// 以项目形态预置状态（测试自检用）：等价于"创建项目"但不弹对话框
    void setProjectForTest(const QString &dir, const QStringList &files);

    /// 同步执行一次完整重建并将结果存入 lastOutput（测试自检用，阻塞直至完成）
    void runPipelineSyncForTest();

    /// 相机位姿 → 金字塔几何（线框 + 底面四角 + 图像路径，显示坐标系 y-up；
    /// depth 为金字塔深度，测试自检用，也供 updateCameraFrustaFromOutput 调用）
    static CameraFrustumGeometry cameraFrustumGeometry(const std::vector<CameraPose> &poses,
                                                       float depth,
                                                       const QStringList &imageFiles);

    /// 仅线框顶点（cameraFrustumGeometry 的便捷封装，测试自检用）
    static QVector<float> cameraFrustumLines(const std::vector<CameraPose> &poses,
                                             float depth);

    /// 最近一次重建输出（测试自检用）
    const ReconstructionOutput &lastOutput() const { return m_pipeline ? m_pipeline->lastOutput() : m_lastFallback; }

private slots:
    void onCreateProject();
    void onOpenProject();
    void onSaveProjectFile();
    void onExportPointCloud();
    void onImportImages();
    void onImportDirectory();

    void onStartReconstruction();
    void onStopReconstruction();
    void onDenseReconstruction();
    void onResetView();
    void onTopView();
    void onFrontView();
    void onSideView();
    void onWireframeMode(bool enabled);
    void onPipelineProgress(const QString &stage, int percent);
    void onPipelineLog(const QString &message);
    void onPipelineLogBatch(const QStringList &messages);
    void onPipelineFinished(bool success);
    void onStagePointCloudReady(const QVector<float> &points,
                                const QVector<float> &colors,
                                const QString &stageName);
    void updateStatusLabel();

private:
    void setupMenuBar();
    void setupStatusBar();
    void setupCentralWidget();
    void connectSignals();
    void loadAndDisplayCloudFromOutput();
    /// 将当前输出的相机位姿叠加为金字塔线框（尺度取点云包围盒的 12%）
    void updateCameraFrustaFromOutput();

    /// 重建运行中时提示并禁止项目操作
    bool ensureProjectOperationAllowed();
    /// 打开项目清单并恢复 UI（失败弹窗提示，返回 false）
    bool openProject(const QString &manifestPath);
    /// 将当前管线配置（算法/图像/项目路径）同步到 pipeline
    void applyConfigToPipeline();
    void updateTitle();
    void markDirty();
    void addToRecentProjects(const QString &manifestPath);
    void updateRecentMenu();
    void closeEvent(QCloseEvent *event) override;

    PluginRegistry *m_registry = nullptr;
    QSplitter *m_splitter = nullptr;
    ConfigPanel *m_configPanel = nullptr;
    PointCloudWidget *m_pointCloudWidget = nullptr;
    LogConsole *m_logConsole = nullptr;
    QLabel *m_statusLabel = nullptr;
    QLabel *m_threadLabel = nullptr;
    QMenu *m_recentMenu = nullptr;
    QAction *m_frustaAction = nullptr;  ///< 视图菜单"显示相机位姿"（与面板复选框同步）

    ReconstructionPipeline *m_pipeline = nullptr;
    PipelineRunner *m_pipelineRunner = nullptr;
    ReconstructionOutput m_lastFallback;  ///< 无 pipeline 时的兜底输出

    // 项目状态
    QString m_projectDir;    ///< 项目根目录（空表示未打开项目）
    QString m_projectName;   ///< 项目名称
    bool m_dirty = false;    ///< 未保存修改标记（标题栏 [*]）
};

#endif // MAINWINDOW_H
