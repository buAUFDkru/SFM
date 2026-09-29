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
#include "ViewWidget/PointCloudWidget.h"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QSplitter>
#include <QStatusBar>
#include <QThread>

#include <opencv2/core.hpp>

#include <algorithm>
#include <limits>

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
    markDirty();
}

void MainWindow::runPipelineSyncForTest()
{
    if (!m_pipeline)
        m_pipeline = new ReconstructionPipeline();   // 注：非 QObject，无父对象
    applyConfigToPipeline();
    ReconstructionConfig cfg = m_pipeline->config();
    // 预设密集方法，避免 execute() 在稀疏阶段后阻塞等待 UI 选择
    cfg.denseReconstructionMethod = QStringLiteral("StereoSGBM");
    cfg.denseReconstructionParams = {
        {QStringLiteral("useWLSFilter"), false},
        {QStringLiteral("numDisparities"), 64},
        {QStringLiteral("blockSize"), 5},
        {QStringLiteral("useHHMode"), false},
        {QStringLiteral("minDisparity"), 0},
        {QStringLiteral("maxDepth"), 50.0},
    };
    m_pipeline->setConfig(cfg);
    ReconstructionOutput output = m_pipeline->execute();

    // 与 UI 路径（onPipelineFinished）保持一致：重建成功后自动保存项目
    if (output.sparseCloud && !output.sparseCloud->empty() && !m_projectDir.isEmpty()) {
        m_pipeline->saveProject();
        m_dirty = false;
        updateTitle();
    }
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
    connect(frusta, &QAction::toggled, this, [this](bool enabled) {
        m_pointCloudWidget->setCameraFrustaVisible(enabled);
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
    m_threadLabel = new QLabel(tr("线程: 0"), this);

    statusBar()->addWidget(m_statusLabel, 1);
    statusBar()->addPermanentWidget(m_threadLabel);
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

    setCentralWidget(m_splitter);
    resize(1260, 760);
}

void MainWindow::connectSignals()
{
    connect(m_configPanel, &ConfigPanel::startRequested,
            this, &MainWindow::onStartReconstruction);
    connect(m_configPanel, &ConfigPanel::stopRequested,
            this, &MainWindow::onStopReconstruction);
    connect(m_configPanel, &ConfigPanel::configChanged,
            this, &MainWindow::markDirty);
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
    m_configPanel->setAlgorithms(m.descriptorType, m.descriptorParams,
                                 m.matcherType, m.matcherParams);
    m_configPanel->setImageFiles(m.imageFiles);
    m_configPanel->setProjectSummary(m_projectName,
                                     tr("%1 张图像").arg(m.imageFiles.size()));
    m_configPanel->setProjectLoaded(true);

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
    cfg.projectPath = m_projectDir;
    cfg.projectName = m_projectName;
    m_pipeline->setConfig(cfg);
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
    m_threadLabel->setText(tr("线程: 1"));
    m_pipelineRunner->start();
}

void MainWindow::onStopReconstruction()
{
    if (m_pipelineRunner)
        m_pipelineRunner->stop();
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
    m_threadLabel->setText(tr("线程: 0"));
    m_statusLabel->setText(success ? tr("重建完成") : tr("重建失败"));
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
}

void MainWindow::updateStatusLabel()
{
    // 保留：均衡信号缺失时刷新状态文本
}

void MainWindow::loadAndDisplayCloudFromOutput()
{
    if (!m_pipeline)
        return;

    const ReconstructionOutput &output = m_pipeline->lastOutput();

    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud =
        (output.denseCloud && !output.denseCloud->empty()) ? output.denseCloud
                                                           : output.sparseCloud;

    if (!cloud || cloud->empty()) {
        m_logConsole->appendInfo(tr("点云为空，无法显示"));
        return;
    }

    QVector<float> points, colors;
    const size_t n = cloud->size();
    const size_t step = (n > 5000000) ? n / 5000000 : 1;
    points.reserve(static_cast<int>(n / step) * 3);
    colors.reserve(static_cast<int>(n / step) * 3);
    for (size_t i = 0; i < n; i += step) {
        const auto &pt = cloud->points[i];
        points.append(pt.x);
        points.append(pt.y);
        points.append(pt.z);
        colors.append(pt.r / 255.0f);
        colors.append(pt.g / 255.0f);
        colors.append(pt.b / 255.0f);
    }
    m_pointCloudWidget->updatePointCloud(points, colors, true);

    updateCameraFrustaFromOutput();
}

CameraFrustumGeometry MainWindow::cameraFrustumGeometry(const std::vector<CameraPose> &poses,
                                                        float depth,
                                                        const QStringList &imageFiles)
{
    CameraFrustumGeometry geom;
    if (depth <= 0.0f)
        return geom;

    for (const auto &pose : poses) {
        if (!pose.registered || pose.R.empty() || pose.t.empty())
            continue;

        // 金字塔底面四角（相机系，光心为原点、看 +z；保持图像宽高比）
        const double hw = 0.5 * depth;
        const double hh = 0.5 * depth *
                          (pose.imageWidth > 0 && pose.imageHeight > 0
                               ? double(pose.imageHeight) / pose.imageWidth
                               : 0.75);
        const double camCorners[4][3] = {
            {-hw, -hh, depth}, {hw, -hh, depth}, {hw, hh, depth}, {-hw, hh, depth}};

        const cv::Mat Rt = pose.R.t();
        const cv::Mat C = -Rt * pose.t;   // 光心（世界系）

        // 显示坐标系转换：OpenCV y-down → y-up（绕 Z 轴 180°，x、y 取反），
        // 与点云组装时的转换一致，否则金字塔与点云错位
        const auto toDisplay = [](const cv::Mat &v) {
            return QVector3D(float(-v.at<double>(0)),
                             float(-v.at<double>(1)),
                             float(v.at<double>(2)));
        };

        const QVector3D center = toDisplay(C);
        QVector3D corners[4];
        for (int i = 0; i < 4; ++i) {
            const cv::Mat v = Rt * (cv::Mat_<double>(3, 1) << camCorners[i][0],
                                    camCorners[i][1], camCorners[i][2]) + C;
            corners[i] = toDisplay(v);
            geom.quadCorners << corners[i];
        }

        const auto appendSegment = [&geom](const QVector3D &a, const QVector3D &b) {
            geom.lines << a.x() << a.y() << a.z() << b.x() << b.y() << b.z();
        };
        // 光心 → 四角（4 条）+ 底面四边（4 条）
        for (int i = 0; i < 4; ++i)
            appendSegment(center, corners[i]);
        for (int i = 0; i < 4; ++i)
            appendSegment(corners[i], corners[(i + 1) % 4]);

        // 图像路径（imageId 越界视为无图）
        geom.imagePaths << (pose.imageId >= 0 && pose.imageId < imageFiles.size()
                                ? imageFiles.at(pose.imageId)
                                : QString());
    }
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
    const ReconstructionOutput &output = m_pipeline->lastOutput();

    // 金字塔深度取点云包围盒最大边长的 12%（无点云时保持单位深度）
    float depth = 1.0f;
    if (output.sparseCloud && !output.sparseCloud->empty()) {
        cv::Vec3d mn(std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max());
        cv::Vec3d mx(-mn[0], -mn[1], -mn[2]);
        for (const auto &pt : output.sparseCloud->points) {
            mn[0] = std::min(mn[0], double(pt.x));
            mn[1] = std::min(mn[1], double(pt.y));
            mn[2] = std::min(mn[2], double(pt.z));
            mx[0] = std::max(mx[0], double(pt.x));
            mx[1] = std::max(mx[1], double(pt.y));
            mx[2] = std::max(mx[2], double(pt.z));
        }
        depth = float(0.12 * std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]}));
        if (depth <= 0.0f)
            depth = 1.0f;
    }

    const QStringList imageFiles = m_pipeline->config().imageFiles;
    const CameraFrustumGeometry geom = cameraFrustumGeometry(output.cameraPoses, depth, imageFiles);
    if (geom.lines.isEmpty()) {
        m_pointCloudWidget->clearCameraFrusta();
        return;
    }

    // 青色线框，与点云形成对比
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
