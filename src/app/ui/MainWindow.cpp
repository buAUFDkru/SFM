/**
 * @file MainWindow.cpp
 * @brief SFM Reconstructor 主窗口实现
 *
 * 界面框架：ConfigPanel | PointCloudWidget | LogConsole 三分栏。
 * 重建在 PipelineRunner 工作线程执行，信号经 QueuedConnection 回到 UI 线程。
 *
 * 项目管理：菜单"文件"提供新建/打开/保存/最近打开；
 * 重建完成后自动保存点云、相机参数、匹配对与清单到项目目录。
 */

#include "MainWindow.h"
#include "ConfigPanel.h"
#include "DenseReconstructionDialog.h"
#include "LogWidget/LogConsole.h"
#include "NewProjectDialog.h"
#include "SparseReconstructionDialog.h"
#include "ViewWidget/PointCloudWidget.h"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QSplitter>
#include <QStatusBar>
#include <QThread>
#include <QToolButton>

#include "pipeline/ReconstructionPipeline.h"
#include "viewmodel/FrustumBuilder.h"

#include <algorithm>
#include <string>
#include <vector>

#include "plugin/PluginRegistry.h"
#include "builtin/registerBuiltIn.h"

namespace {
constexpr char kRecentProjectsKey[] = "recentProjects";
constexpr int kMaxRecentProjects = 8;
} // namespace

MainWindow::MainWindow(const QString &pluginsDir, QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("SFM Reconstructor v3.0 — 特征重建"));

    // 插件注册表：先注册内置算法，再扫描 DLL 插件（供 ConfigPanel 算法下拉）
    m_registry = &PluginRegistry::instance();
    registerBuiltInAlgorithms(*m_registry);
    m_registry->scanPlugins(pluginsDir);

    setupMenuBar();
    setupStatusBar();
    setupCentralWidget();
    connectSignals();
    updateRecentMenu();
}

MainWindow::~MainWindow()
{
    onStopReconstruction();
    delete m_pipelineRunner;
    delete m_pipeline;
}

PluginRegistry *MainWindow::registry() const
{
    return m_registry;
}

QStringList MainWindow::currentImageFiles() const
{
    return m_configPanel ? m_configPanel->gatherConfig().imageFiles : QStringList();
}

int MainWindow::currentImageCount() const
{
    return m_configPanel ? m_configPanel->imageCount() : 0;
}

void MainWindow::setProjectForTest(const QString &dir, const QStringList &files)
{
    m_projectDir = dir;
    m_projectName = QFileInfo(dir).fileName();
    m_configPanel->setProjectSummary(m_projectName,
                                     tr("%1 张图像").arg(files.size()));
    m_configPanel->setProjectLoaded(true);
    m_configPanel->setImageFiles(files);
    m_sparseDialog->setStartEnabled(true);
    markDirty();
}

void MainWindow::runPipelineSyncForTest()
{
    if (!m_pipeline)
        m_pipeline = new ReconstructionPipeline();   // 注：非 QObject，无父对象
    applyConfigToPipeline();
    ReconstructionConfig cfg = m_pipeline->config();
    // 预设密集方法，避免 execute() 在稀疏阶段后阻塞等待 UI 选择；
    // PatchMatchV2 快速参数保证测试时长可控
    cfg.denseReconstructionMethod = QStringLiteral("PatchMatchV2");
    cfg.denseReconstructionParams = {
        {QStringLiteral("maxImageSize"), 320},
        {QStringLiteral("windowRadius"), 3},
        {QStringLiteral("numSamples"), 5},
        {QStringLiteral("numIterations"), 2},
        {QStringLiteral("refineIterations"), 1},
        {QStringLiteral("multiscale"), false},
        {QStringLiteral("costTopK"), 1},
        {QStringLiteral("geomConsistency"), false},
        {QStringLiteral("fusion"), false},
        {QStringLiteral("planePrior"), false},
        {QStringLiteral("filter"), false},
    };
    m_pipeline->setConfig(cfg);
    m_pipeline->execute();

    // 与 UI 路径（onPipelineFinished）保持一致：重建成功后自动保存项目，
    // 并完成点云显示与显示选项复选框同步（测试直调路径无回调事件）
    if (m_pipeline->hasCloud(false) && !m_projectDir.isEmpty()) {
        m_pipeline->saveProject();
        m_dirty = false;
        updateTitle();
    }
    if (m_pipeline->hasCloud(false))
        loadAndDisplayCloudFromOutput();
}

void MainWindow::setupMenuBar()
{
    // ── 文件 ──
    QMenu *fileMenu = menuBar()->addMenu(tr("文件(&F)"));

    QAction *newProj = fileMenu->addAction(tr("新建项目(&N)..."));
    newProj->setShortcut(QKeySequence::New);
    connect(newProj, &QAction::triggered, this, &MainWindow::onCreateProject);

    QAction *openProj = fileMenu->addAction(tr("打开项目(&O)..."));
    openProj->setShortcut(QKeySequence::Open);
    connect(openProj, &QAction::triggered, this, &MainWindow::onOpenProject);

    m_recentMenu = fileMenu->addMenu(tr("最近打开(&T)"));

    fileMenu->addSeparator();

    QAction *importImg = fileMenu->addAction(tr("导入图像(&I)..."));
    importImg->setShortcut(QKeySequence("Ctrl+I"));
    connect(importImg, &QAction::triggered, this, &MainWindow::onImportImages);

    QAction *importDir = fileMenu->addAction(tr("导入图像目录(&D)..."));
    connect(importDir, &QAction::triggered, this, &MainWindow::onImportDirectory);

    fileMenu->addSeparator();

    QAction *saveProj = fileMenu->addAction(tr("保存项目(&S)"));
    saveProj->setShortcut(QKeySequence::Save);
    connect(saveProj, &QAction::triggered, this, &MainWindow::onSaveProjectFile);

    QAction *exportPc = fileMenu->addAction(tr("导出点云(&E)..."));
    exportPc->setShortcut(QKeySequence("Ctrl+E"));
    connect(exportPc, &QAction::triggered, this, &MainWindow::onExportPointCloud);

    fileMenu->addSeparator();

    QAction *exit = fileMenu->addAction(tr("退出(&X)"));
    exit->setShortcut(QKeySequence("Alt+F4"));
    connect(exit, &QAction::triggered, this, &QMainWindow::close);

    // ── 重建 ──
    QMenu *reconstructMenu = menuBar()->addMenu(tr("重建(&R)"));
    QAction *sparse = reconstructMenu->addAction(tr("稀疏重建(&S)..."));
    sparse->setShortcut(QKeySequence("Ctrl+R"));
    connect(sparse, &QAction::triggered, this, &MainWindow::onShowSparseReconstruction);

    QAction *dense = reconstructMenu->addAction(tr("密集重建(&D)..."));
    connect(dense, &QAction::triggered, this, &MainWindow::onDenseReconstruction);

    // ── 视图 ──
    QMenu *viewMenu = menuBar()->addMenu(tr("视图(&V)"));
    QAction *resetView = viewMenu->addAction(tr("重置视角(&R)"));
    resetView->setShortcut(QKeySequence("R"));
    connect(resetView, &QAction::triggered, this, &MainWindow::onResetView);

    QAction *topView = viewMenu->addAction(tr("顶视图(&T)"));
    topView->setShortcut(QKeySequence("T"));
    connect(topView, &QAction::triggered, this, &MainWindow::onTopView);

    QAction *frontView = viewMenu->addAction(tr("前视图(&F)"));
    connect(frontView, &QAction::triggered, this, &MainWindow::onFrontView);

    QAction *sideView = viewMenu->addAction(tr("侧视图(&L)"));
    connect(sideView, &QAction::triggered, this, &MainWindow::onSideView);

    viewMenu->addSeparator();

    QAction *wireframe = viewMenu->addAction(tr("线框模式(&W)"));
    wireframe->setCheckable(true);
    connect(wireframe, &QAction::toggled, this, &MainWindow::onWireframeMode);

    QAction *frusta = viewMenu->addAction(tr("显示相机位姿(&P)"));
    frusta->setCheckable(true);
    frusta->setChecked(true);
    m_frustaAction = frusta;
    connect(frusta, &QAction::toggled, this, [this](bool enabled) {
        m_pointCloudWidget->setCameraFrustaVisible(enabled);
        m_configPanel->setCameraFrustaChecked(enabled);
    });

    // ── 帮助 ──
    QMenu *helpMenu = menuBar()->addMenu(tr("帮助(&H)"));
    QAction *about = helpMenu->addAction(tr("关于(&A)"));
    connect(about, &QAction::triggered, this, [this]() {
        QMessageBox::about(this, tr("关于 SFM Reconstructor"),
                           tr("<h2>SFM Reconstructor v3.0</h2>"
                              "<p>基于 C++/Qt 的桌面端三维重建工具（插件化重构）</p>"
                              "<p>技术栈: Qt 6 | OpenCV | PCL | OpenGL | Ceres | Open3D</p>"));
    });
}

void MainWindow::setupStatusBar()
{
    m_statusLabel = new QLabel(tr("就绪"), this);
    statusBar()->addWidget(m_statusLabel, 1);
}

void MainWindow::setupCentralWidget()
{
    m_configPanel = new ConfigPanel(m_registry, this);
    m_configPanel->setMinimumWidth(320);
    m_configPanel->setMaximumWidth(560);

    m_pointCloudWidget = new PointCloudWidget(this);
    m_pointCloudWidget->setMinimumSize(640, 480);

    m_logConsole = new LogConsole(this);
    m_logConsole->setMinimumWidth(260);
    m_logConsole->setMaximumWidth(520);

    m_splitter = new QSplitter(Qt::Horizontal, this);
    m_splitter->addWidget(m_configPanel);
    m_splitter->addWidget(m_pointCloudWidget);
    m_splitter->addWidget(m_logConsole);

    m_splitter->setStretchFactor(0, 0);
    m_splitter->setStretchFactor(1, 1);
    m_splitter->setStretchFactor(2, 0);
    m_splitter->setSizes({380, 920, 300});

    // 稀疏重建对话框（非模态，经菜单"重建 → 稀疏重建"打开）
    m_sparseDialog = new SparseReconstructionDialog(m_registry, this);

    // 最右侧日志收展按钮：竖向高亮条，点击隐藏/显示日志面板
    m_logToggle = new QToolButton(this);
    m_logToggle->setToolButtonStyle(Qt::ToolButtonTextOnly);
    QFont toggleFont = m_logToggle->font();
    toggleFont.setPointSize(11);
    toggleFont.setBold(true);
    m_logToggle->setFont(toggleFont);
    m_logToggle->setText(tr("▶"));   // 日志显示中：点击向右收起
    m_logToggle->setToolTip(tr("隐藏日志"));
    m_logToggle->setFixedWidth(26);
    m_logToggle->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    m_logToggle->setStyleSheet(
        QStringLiteral(
            "QToolButton { background: #3d4450; border: none; border-left: 1px solid #5a626e;"
            "  color: #e0e0e0; padding: 0; }"
            "QToolButton:hover { background: #4a90d9; }"
            "QToolButton:pressed { background: #3a7bbf; }"));
    connect(m_logToggle, &QToolButton::clicked,
            this, &MainWindow::onToggleLogPanel);

    auto *central = new QWidget(this);
    auto *centralLayout = new QHBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->setSpacing(0);
    centralLayout->addWidget(m_splitter, 1);
    centralLayout->addWidget(m_logToggle);
    setCentralWidget(central);

    resize(1260, 760);
}

void MainWindow::connectSignals()
{
    connect(m_sparseDialog, &SparseReconstructionDialog::startRequested,
            this, &MainWindow::onStartReconstruction);
    connect(m_sparseDialog, &SparseReconstructionDialog::stopRequested,
            this, &MainWindow::onStopReconstruction);
    connect(m_sparseDialog, &SparseReconstructionDialog::configChanged,
            this, &MainWindow::markDirty);
    connect(m_configPanel, &ConfigPanel::configChanged,
            this, &MainWindow::markDirty);
    connect(m_configPanel, &ConfigPanel::cameraFrustaVisibleChanged,
            this, [this](bool visible) {
                m_pointCloudWidget->setCameraFrustaVisible(visible);
                if (m_frustaAction)
                    m_frustaAction->setChecked(visible);
            });
    // 稀疏/密集点云互斥显示：勾选一个时显示对应点云，
    // 并在此信号处理中联动取消另一个（非 radio 语义）
    connect(m_configPanel, &ConfigPanel::sparseCloudVisibleChanged,
            this, &MainWindow::onSparseCloudVisible);
    connect(m_configPanel, &ConfigPanel::denseCloudVisibleChanged,
            this, &MainWindow::onDenseCloudVisible);
}

// ═══════════════════════════ 项目管理 ═══════════════════════════

bool MainWindow::ensureProjectOperationAllowed()
{
    if (m_pipelineRunner && m_pipelineRunner->isRunning()) {
        QMessageBox::information(this, tr("项目"),
                                 tr("重建正在进行中，请等待完成或停止后再操作。"));
        return false;
    }
    return true;
}

void MainWindow::onCreateProject()
{
    if (!ensureProjectOperationAllowed())
        return;

    NewProjectDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted)
        return;

    // 释放旧管线状态（新建项目从零开始）
    onStopReconstruction();
    delete m_pipelineRunner;
    m_pipelineRunner = nullptr;
    delete m_pipeline;
    m_pipeline = nullptr;

    const QString dir = dlg.projectDir();
    QDir().mkpath(dir);

    m_projectDir = dir;
    m_projectName = dlg.projectName();

    if (!m_configPanel) {
        m_logConsole->appendError(tr("界面尚未初始化"));
        return;
    }
    m_configPanel->setImageFiles(dlg.imageFiles());  // 自动检测 cameramatrix.txt
    m_configPanel->setProjectSummary(m_projectName,
                                     tr("%1 张图像").arg(dlg.imageFiles().size()));
    m_configPanel->setProjectLoaded(true);
    // 新项目尚无点云：禁用并复位稀疏/密集显示选项
    m_configPanel->setCloudAvailability(false, false);
    m_sparseDialog->setStartEnabled(true);

    markDirty();
    m_logConsole->appendInfo(tr("[项目] 已创建: %1（%2 张图像）")
                                 .arg(dir).arg(dlg.imageFiles().size()));

    // 新项目从空场景开始
    m_pointCloudWidget->clearPoints();
    m_pointCloudWidget->clearCameraFrusta();
}

void MainWindow::onOpenProject()
{
    if (!ensureProjectOperationAllowed())
        return;

    const QString path = QFileDialog::getOpenFileName(
        this, tr("打开项目"), QString(),
        tr("SFM 项目 (*.sfmproj);;所有文件 (*)"));
    if (path.isEmpty())
        return;

    openProject(path);
}

bool MainWindow::openProject(const QString &manifestPath)
{
    onStopReconstruction();
    delete m_pipelineRunner;
    m_pipelineRunner = nullptr;

    if (!m_pipeline)
        m_pipeline = new ReconstructionPipeline();

    QString error;
    if (!m_pipeline->loadProject(manifestPath, &error)) {
        QMessageBox::warning(this, tr("打开项目"), tr("打开项目失败：\n%1").arg(error));
        return false;
    }

    const ProjectManifest &m = m_pipeline->manifest();
    m_projectDir = QFileInfo(manifestPath).absolutePath();
    m_projectName = m.name;

    // 恢复 UI：先恢复固定内参，再设置图像列表（其内自动检测会被固定内参覆盖保护）
    const ReconstructionConfig &cfg = m_pipeline->config();
    m_configPanel->setFixedCameraMatrix(cfg.cameraMatrix, cfg.fixCameraMatrix);
    m_sparseDialog->setAlgorithms(m.descriptorType, m.descriptorParams,
                                  m.matcherType, m.matcherParams);
    m_configPanel->setImageFiles(m.imageFiles);
    m_configPanel->setProjectSummary(m_projectName,
                                     tr("%1 张图像").arg(m.imageFiles.size()));
    m_configPanel->setProjectLoaded(true);
    m_sparseDialog->setStartEnabled(true);

    // 图像缺失检查：影响完整重建，不影响查看与密集续算
    QStringList missing;
    for (const QString &f : m.imageFiles) {
        if (!QFile::exists(f))
            missing << f;
    }
    if (!missing.isEmpty()) {
        QMessageBox::warning(this, tr("打开项目"),
                             tr("%1 张图像不存在（已被移动或删除），无法重新执行完整重建。\n"
                                "例如：%2")
                                 .arg(missing.size())
                                 .arg(missing.first()));
        m_logConsole->appendWarning(
            tr("[项目] %1 张图像路径失效，完整重建不可用").arg(missing.size()));
    }

    // 显示已保存的点云（稠密优先，其次稀疏）
    loadAndDisplayCloudFromOutput();

    addToRecentProjects(QFileInfo(manifestPath).absoluteFilePath());
    m_dirty = false;
    updateTitle();

    m_logConsole->appendInfo(tr("[项目] 已打开: %1").arg(m_projectDir));
    return true;
}

void MainWindow::onSaveProjectFile()
{
    if (!ensureProjectOperationAllowed())
        return;

    if (m_projectDir.isEmpty() || !m_pipeline) {
        QMessageBox::information(this, tr("保存项目"), tr("未打开项目。"));
        return;
    }

    applyConfigToPipeline();  // 同步面板上最新的算法参数与图像列表
    QString error;
    if (m_pipeline->saveProject(&error)) {
        m_dirty = false;
        updateTitle();
        m_logConsole->appendInfo(tr("[项目] 已保存: %1").arg(m_projectDir));
    } else {
        QMessageBox::warning(this, tr("保存项目"), tr("保存失败：\n%1").arg(error));
    }
}

void MainWindow::updateTitle()
{
    if (m_projectName.isEmpty()) {
        setWindowTitle(QStringLiteral("SFM Reconstructor v3.0 — 特征重建"));
        setWindowModified(false);
    } else {
        setWindowTitle(QStringLiteral("%1 — SFM Reconstructor v3.0[*]")
                           .arg(m_projectName));
        setWindowModified(m_dirty);
    }
}

void MainWindow::markDirty()
{
    if (!m_dirty) {
        m_dirty = true;
        updateTitle();
    }
}

void MainWindow::addToRecentProjects(const QString &manifestPath)
{
    QSettings settings;
    QStringList list = settings.value(QLatin1String(kRecentProjectsKey)).toStringList();
    list.removeAll(manifestPath);
    list.prepend(manifestPath);
    while (list.size() > kMaxRecentProjects)
        list.removeLast();
    settings.setValue(QLatin1String(kRecentProjectsKey), list);
    updateRecentMenu();
}

void MainWindow::updateRecentMenu()
{
    if (!m_recentMenu)
        return;
    m_recentMenu->clear();

    QSettings settings;
    const QStringList list =
        settings.value(QLatin1String(kRecentProjectsKey)).toStringList();
    for (const QString &path : list) {
        if (!QFileInfo::exists(path))
            continue;
        QAction *act = m_recentMenu->addAction(
            QFileInfo(path).completeBaseName());
        act->setToolTip(path);
        const QString p = path;
        connect(act, &QAction::triggered, this, [this, p]() {
            if (ensureProjectOperationAllowed())
                openProject(p);
        });
    }
    if (m_recentMenu->isEmpty())
        m_recentMenu->addAction(tr("（空）"))->setEnabled(false);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (m_dirty && !m_projectDir.isEmpty()) {
        const QMessageBox::StandardButton ret = QMessageBox::question(
            this, tr("关闭"),
            tr("项目“%1”有未保存的修改，是否保存？").arg(m_projectName),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
        if (ret == QMessageBox::Cancel) {
            event->ignore();
            return;
        }
        if (ret == QMessageBox::Save) {
            applyConfigToPipeline();
            QString error;
            if (!m_pipeline || !m_pipeline->saveProject(&error)) {
                QMessageBox::warning(this, tr("保存项目"),
                                     tr("保存失败：\n%1").arg(error));
                event->ignore();
                return;
            }
        }
    }
    event->accept();
}

// ═══════════════════════════ 重建控制 ═══════════════════════════

void MainWindow::applyConfigToPipeline()
{
    if (!m_pipeline)
        return;
    ReconstructionConfig cfg = m_configPanel->gatherConfig();
    m_sparseDialog->gatherAlgorithms(cfg.descriptorType, cfg.descriptorParams,
                                     cfg.matcherType, cfg.matcherParams);
    cfg.projectPath = m_projectDir;
    cfg.projectName = m_projectName;
    m_pipeline->setConfig(cfg);
}

void MainWindow::onShowSparseReconstruction()
{
    m_sparseDialog->show();
    m_sparseDialog->raise();
    m_sparseDialog->activateWindow();
}

void MainWindow::onToggleLogPanel()
{
    const bool showing = m_logConsole->isVisible();
    const QList<int> sizes = m_splitter->sizes();
    const int panel = sizes.value(0, 380);   // 左侧面板宽度保持不变

    if (showing) {
        // 收起：日志宽度全部让给中间视图
        if (sizes.size() == 3 && sizes[2] > 0)
            m_logWidth = sizes[2];
        m_logConsole->hide();
        m_splitter->setSizes({panel, sizes.value(1) + m_logWidth});
        m_logToggle->setText(tr("◀"));
        m_logToggle->setToolTip(tr("显示日志"));
    } else {
        // 展开：恢复日志宽度，中间视图让出相应空间
        const int logW = m_logWidth > 0 ? m_logWidth : 300;
        m_logConsole->show();
        m_splitter->setSizes({panel,
                              qMax(m_pointCloudWidget->minimumWidth(),
                                   sizes.value(1, 600) - logW),
                              logW});
        m_logToggle->setText(tr("▶"));
        m_logToggle->setToolTip(tr("隐藏日志"));
    }
}

void MainWindow::onStartReconstruction()
{
    if (m_pipelineRunner && m_pipelineRunner->isRunning())
        return;

    if (m_projectDir.isEmpty()) {
        QMessageBox::information(this, tr("重建"), tr("请先创建或打开项目。"));
        return;
    }

    onStopReconstruction();
    delete m_pipelineRunner;
    m_pipelineRunner = nullptr;
    delete m_pipeline;
    m_pipeline = nullptr;

    m_pipeline = new ReconstructionPipeline();   // 注：非 QObject，无父对象
    applyConfigToPipeline();

    m_pipelineRunner = new PipelineRunner(m_pipeline, this);

    connect(m_pipelineRunner, &PipelineRunner::progressChanged,
            this, &MainWindow::onPipelineProgress);
    connect(m_pipelineRunner, &PipelineRunner::logEmitted,
            this, &MainWindow::onPipelineLog);
    connect(m_pipelineRunner, &PipelineRunner::logBatchEmitted,
            this, &MainWindow::onPipelineLogBatch);
    connect(m_pipelineRunner, &PipelineRunner::stagePointCloudReady,
            this, &MainWindow::onStagePointCloudReady);
    connect(m_pipelineRunner, &PipelineRunner::denseReconstructionMethodRequested,
            this, [this]() {
                DenseReconstructionDialog dialog(this);
                if (dialog.exec() == QDialog::Accepted) {
                    m_pipelineRunner->setDenseReconstructionMethod(
                        dialog.selectedMethod(), dialog.selectedParams());
                } else {
                    m_pipelineRunner->stop();
                }
            });
    connect(m_pipelineRunner, &PipelineRunner::finished,
            this, &MainWindow::onPipelineFinished);

    m_statusLabel->setText(tr("重建中..."));
    m_sparseDialog->setRunning(true);
    m_pipelineRunner->start();
}

void MainWindow::onStopReconstruction()
{
    if (m_pipelineRunner)
        m_pipelineRunner->stop();
    m_sparseDialog->setRunning(false);
    m_statusLabel->setText(tr("已停止"));
}

void MainWindow::onDenseReconstruction()
{
    if (!m_pipeline || !m_pipeline->denseReconstructionAvailable()) {
        QMessageBox::information(this, tr("密集重建"),
                                 tr("请先完成稀疏重建，或等待当前重建结束。"));
        return;
    }
    if (!m_pipelineRunner || m_pipelineRunner->isRunning()) {
        QMessageBox::information(this, tr("密集重建"), tr("重建正在运行中，请等待完成。"));
        return;
    }

    DenseReconstructionDialog dialog(this);
    if (dialog.exec() != QDialog::Accepted)
        return;

    ReconstructionConfig config = m_pipeline->config();
    config.denseReconstructionMethod = dialog.selectedMethod();
    config.denseReconstructionParams = dialog.selectedParams();
    m_pipeline->setConfig(config);

    if (config.denseReconstructionMethod.isEmpty()) {
        QMessageBox::warning(this, tr("密集重建"), tr("未选择密集重建方法。"));
        return;
    }

    m_statusLabel->setText(tr("密集重建: %1").arg(config.denseReconstructionMethod));
    m_sparseDialog->setRunning(true);
    m_pipelineRunner->startDenseReconstruction();
}

void MainWindow::onExportPointCloud()
{
    const QString filePath = QFileDialog::getSaveFileName(
        this, tr("导出点云"), QString(), tr("PLY 文件 (*.ply);;所有文件 (*)"));
    if (filePath.isEmpty())
        return;
    if (m_pipeline) {
        m_pipeline->exportPointCloud(filePath);
        m_logConsole->appendInfo(tr("点云已导出: %1").arg(filePath));
    }
}

void MainWindow::onImportImages()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, tr("导入图像"), QString(),
        tr("图像 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"));
    if (!files.isEmpty())
        m_configPanel->setImageFiles(files);
}

void MainWindow::onImportDirectory()
{
    const QString dir = QFileDialog::getExistingDirectory(this, tr("导入图像目录"));
    if (!dir.isEmpty())
        m_configPanel->setImageDirectory(dir);
}

// ═══════════════════════════ 管线回调 ═══════════════════════════

void MainWindow::onPipelineProgress(const QString &stage, int percent)
{
    m_statusLabel->setText(tr("%1: %2%").arg(stage).arg(percent));
}

void MainWindow::onPipelineLog(const QString &message)
{
    m_logConsole->appendInfo(message);
}

void MainWindow::onPipelineLogBatch(const QStringList &messages)
{
    m_logConsole->appendBatch(messages);
}

void MainWindow::onPipelineFinished(bool success)
{
    m_statusLabel->setText(success ? tr("重建完成") : tr("重建失败"));
    m_sparseDialog->setRunning(false);
    if (!success)
        return;

    loadAndDisplayCloudFromOutput();

    // 自动保存项目结果（点云 + 相机参数 + 匹配对 + 清单）
    if (!m_projectDir.isEmpty() && m_pipeline) {
        QString error;
        if (m_pipeline->saveProject(&error)) {
            m_dirty = false;
            updateTitle();
            m_logConsole->appendInfo(tr("[项目] 结果已自动保存: %1").arg(m_projectDir));
        } else {
            m_logConsole->appendError(
                tr("[项目] 自动保存失败: %1").arg(error));
        }
    }
}

void MainWindow::onStagePointCloudReady(const QVector<float> &points,
                                        const QVector<float> &colors,
                                        const QString &stageName)
{
    m_logConsole->appendInfo(tr("[%1] 阶段点云: %2 点")
                                 .arg(stageName).arg(points.size() / 3));
    m_pointCloudWidget->updatePointCloud(points, colors, true);
    m_showDense = (stageName == QStringLiteral("dense"));

    // 阶段点云就绪即同步：稀疏阶段显示稀疏、密集阶段显示密集，
    // 显示选项复选框跟随切换（重建流程：稀疏先生成后密集覆盖显示）。
    // 注意 dense 阶段事件发射于 lastOutput.denseCloud 组装之前，
    // 存在性按事件本身推断，不能依赖 cloudAvailable
    if (m_configPanel) {
        m_configPanel->setCloudAvailability(m_showDense ? cloudAvailable(false)
                                                        : true,
                                            m_showDense);
        m_configPanel->setSparseCloudChecked(!m_showDense);
        m_configPanel->setDenseCloudChecked(m_showDense);
    }

    // 阶段点云就绪即同步相机位姿金字塔（稀疏完成时 lastOutput 已含位姿）
    updateCameraFrustaFromOutput();
}

void MainWindow::onSparseCloudVisible(bool checked)
{
    if (!checked)           // 取消勾选不改变当前显示
        return;
    if (!cloudAvailable(false)) {
        m_logConsole->appendInfo(tr("稀疏点云不可用"));
        return;
    }
    m_showDense = false;
    displayCloudFromOutput(false);
    if (m_configPanel)
        m_configPanel->setDenseCloudChecked(false);   // 信号联动关闭另一个
}

void MainWindow::onDenseCloudVisible(bool checked)
{
    if (!checked)
        return;
    if (!cloudAvailable(true)) {
        m_logConsole->appendInfo(tr("密集点云不可用"));
        return;
    }
    m_showDense = true;
    displayCloudFromOutput(true);
    if (m_configPanel)
        m_configPanel->setSparseCloudChecked(false);  // 信号联动关闭另一个
}

void MainWindow::updateStatusLabel()
{
    // 保留：均衡信号缺失时刷新状态文本
}

void MainWindow::loadAndDisplayCloudFromOutput()
{
    // 加载/重建完成：优先显示密集点云（不存在时显示稀疏）
    const bool dense = cloudAvailable(true);
    m_showDense = dense;
    displayCloudFromOutput(dense);
    if (m_configPanel)
        syncCloudDisplayChecks(dense);
    updateCameraFrustaFromOutput();
}

bool MainWindow::cloudAvailable(bool dense) const
{
    return m_pipeline && m_pipeline->hasCloud(dense);
}

void MainWindow::displayCloudFromOutput(bool dense)
{
    if (!m_pipeline)
        return;
    const sfm::view::ReconstructionView view = m_pipeline->lastView();
    const sfm::view::PointCloudView &cloud = dense ? view.dense : view.sparse;

    if (cloud.empty()) {
        m_logConsole->appendInfo(tr("点云为空，无法显示"));
        return;
    }

    const QVector<float> points(cloud.positions.begin(), cloud.positions.end());
    const QVector<float> colors(cloud.colors.begin(), cloud.colors.end());
    m_pointCloudWidget->updatePointCloud(points, colors, true);
}

sfm::view::ReconstructionView MainWindow::lastView() const
{
    if (!m_pipeline)
        return {};
    return m_pipeline->lastView();
}

void MainWindow::syncCloudDisplayChecks(bool denseSelected)
{
    if (!m_configPanel)
        return;
    const bool hasSparse = cloudAvailable(false);
    const bool hasDense = cloudAvailable(true);
    // 先按存在性启用，再同步勾选（静默，不触发显示切换信号）
    m_configPanel->setCloudAvailability(hasSparse, hasDense);
    m_configPanel->setSparseCloudChecked(!denseSelected && hasSparse);
    m_configPanel->setDenseCloudChecked(denseSelected && hasDense);
}

CameraFrustumGeometry MainWindow::cameraFrustumGeometry(const std::vector<CameraPose> &poses,
                                                        float depth,
                                                        const QStringList &imageFiles)
{
    std::vector<std::string> paths;
    paths.reserve(static_cast<size_t>(imageFiles.size()));
    for (const QString &f : imageFiles)
        paths.push_back(f.toStdString());

    const sfm::view::CameraFrustumView v =
        sfm::view::buildCameraFrusta(poses, depth, paths);

    CameraFrustumGeometry geom;
    geom.lines = QVector<float>(v.lineSegments.begin(), v.lineSegments.end());
    geom.quadCorners.reserve(static_cast<int>(v.quadCorners.size() / 3));
    for (size_t i = 0; i + 2 < v.quadCorners.size(); i += 3) {
        geom.quadCorners.append(QVector3D(v.quadCorners[i],
                                          v.quadCorners[i + 1],
                                          v.quadCorners[i + 2]));
    }
    for (const auto &p : v.imagePaths)
        geom.imagePaths.append(QString::fromStdString(p));
    return geom;
}

QVector<float> MainWindow::cameraFrustumLines(const std::vector<CameraPose> &poses,
                                              float depth)
{
    return cameraFrustumGeometry(poses, depth, {}).lines;
}

void MainWindow::updateCameraFrustaFromOutput()
{
    if (!m_pipeline)
        return;
    const sfm::view::ReconstructionView view = m_pipeline->lastView();
    if (view.frusta.lineSegments.empty()) {
        m_pointCloudWidget->clearCameraFrusta();
        return;
    }

    CameraFrustumGeometry geom;
    geom.lines = QVector<float>(view.frusta.lineSegments.begin(),
                                 view.frusta.lineSegments.end());
    for (size_t i = 0; i + 2 < view.frusta.quadCorners.size(); i += 3) {
        geom.quadCorners.append(QVector3D(view.frusta.quadCorners[i],
                                          view.frusta.quadCorners[i + 1],
                                          view.frusta.quadCorners[i + 2]));
    }
    for (const auto &p : view.frusta.imagePaths)
        geom.imagePaths.append(QString::fromStdString(p));

    QVector<float> frustaColors;
    frustaColors.reserve(geom.lines.size());
    for (int i = 0; i < geom.lines.size() / 3; ++i)
        frustaColors << 0.30f << 0.85f << 1.00f;
    m_pointCloudWidget->setCameraFrusta(geom.lines, frustaColors,
                                        geom.quadCorners, geom.imagePaths);
}

void MainWindow::onResetView()
{
    m_pointCloudWidget->resetView();
}

void MainWindow::onTopView()
{
    m_pointCloudWidget->setTopView();
}

void MainWindow::onFrontView()
{
    m_pointCloudWidget->setFrontView();
}

void MainWindow::onSideView()
{
    m_pointCloudWidget->setSideView();
}

void MainWindow::onWireframeMode(bool enabled)
{
    m_pointCloudWidget->setWireframeMode(enabled);
}
