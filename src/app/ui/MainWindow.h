#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <QVector3D>

#include "data/SfmTypes.h"
#include "pipeline/PipelineRunner.h"
#include "viewmodel/RenderModel.h"

class ConfigPanel;
class LogConsole;
class PointCloudWidget;
class PluginRegistry;
class ReconstructionPipeline;
class QAction;
class QLabel;
class QMenu;
class QSplitter;
class SparseReconstructionDialog;
class QToolButton;

/// 相机金字塔几何数据（线框 + 底面四角 + 对应图像路径）
struct CameraFrustumGeometry
{
    QVector<float> lines;               ///< 线框成对端点（显示坐标系）
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

    /// 同步执行一次完整重建并将结果存入 lastView（测试自检用，阻塞直至完成）
    void runPipelineSyncForTest();

    /// 相机位姿 → 金字塔几何（委托 FrustumBuilder；测试自检用）
    static CameraFrustumGeometry cameraFrustumGeometry(const std::vector<CameraPose> &poses,
                                                       float depth,
                                                       const QStringList &imageFiles);

    /// 仅线框顶点（cameraFrustumGeometry 的便捷封装，测试自检用）
    static QVector<float> cameraFrustumLines(const std::vector<CameraPose> &poses,
                                             float depth);

    /// 最近一次重建的渲染视图（无 PCL；测试自检用）
    sfm::view::ReconstructionView lastView() const;

    /// 内部管线（测试需要 PCL 级断言时使用）
    ReconstructionPipeline *pipeline() const { return m_pipeline; }

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
    void onShowSparseReconstruction();
    void onToggleLogPanel();
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
    void onSparseCloudVisible(bool checked);
    void onDenseCloudVisible(bool checked);
    void updateStatusLabel();

private:
    void setupMenuBar();
    void setupStatusBar();
    void setupCentralWidget();
    void connectSignals();
    void loadAndDisplayCloudFromOutput();
    void displayCloudFromOutput(bool dense);
    void syncCloudDisplayChecks(bool denseSelected);
    bool cloudAvailable(bool dense) const;
    void updateCameraFrustaFromOutput();

    bool ensureProjectOperationAllowed();
    bool openProject(const QString &manifestPath);
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
    SparseReconstructionDialog *m_sparseDialog = nullptr;
    QToolButton *m_logToggle = nullptr;
    int m_logWidth = 300;
    bool m_showDense = false;
    QLabel *m_statusLabel = nullptr;
    QMenu *m_recentMenu = nullptr;
    QAction *m_frustaAction = nullptr;

    ReconstructionPipeline *m_pipeline = nullptr;
    PipelineRunner *m_pipelineRunner = nullptr;

    QString m_projectDir;
    QString m_projectName;
    bool m_dirty = false;
};

#endif // MAINWINDOW_H
