/**
 * @file MainWindow.cpp
 * @brief SFM Reconstructor 主窗口实现
 *
 * 界面框架：ConfigPanel | PointCloudWidget | LogConsole 三分栏。
 * 重建在 PipelineRunner 工作线程执行，信号经 QueuedConnection 回到 UI 线程。
 */

#include "MainWindow.h"
#include "ConfigPanel.h"
#include "DenseReconstructionDialog.h"
#include "LogWidget/LogConsole.h"
#include "ViewWidget/PointCloudWidget.h"

#include <QAction>
#include <QApplication>
#include <QCoreApplication>
#include <QFileDialog>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QSplitter>
#include <QStatusBar>
#include <QThread>

#include "plugin/PluginRegistry.h"

MainWindow::MainWindow(const QString &pluginsDir, QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("SFM Reconstructor v3.0 — 特征重建"));

    // 插件注册表（扫描后供 ConfigPanel 算法下拉）
    m_registry = &PluginRegistry::instance();
    m_registry->scanPlugins(pluginsDir);

    setupMenuBar();
    setupStatusBar();
    setupCentralWidget();
    connectSignals();
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

void MainWindow::setImageFilesForTest(const QStringList &files)
{
    if (m_configPanel)
        m_configPanel->setImageFiles(files);
}

void MainWindow::setProjectPathForTest(const QString &path)
{
    if (m_configPanel)
        m_configPanel->setProjectPath(path);
}

void MainWindow::runPipelineSyncForTest()
{
    if (!m_pipeline)
        m_pipeline = new ReconstructionPipeline();   // 注：非 QObject，无父对象
    ReconstructionConfig cfg = m_configPanel->gatherConfig();
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
    m_pipeline->execute();
}

void MainWindow::setupMenuBar()
{
    // ── 文件 ──
    QMenu *fileMenu = menuBar()->addMenu(tr("文件(&F)"));

    QAction *importImg = fileMenu->addAction(tr("导入图像(&I)..."));
    importImg->setShortcut(QKeySequence("Ctrl+I"));
    connect(importImg, &QAction::triggered, this, &MainWindow::onImportImages);

    QAction *importDir = fileMenu->addAction(tr("导入图像目录(&D)..."));
    connect(importDir, &QAction::triggered, this, &MainWindow::onImportDirectory);

    fileMenu->addSeparator();

    QAction *exportPc = fileMenu->addAction(tr("导出点云(&E)..."));
    exportPc->setShortcut(QKeySequence("Ctrl+E"));
    connect(exportPc, &QAction::triggered, this, &MainWindow::onExportPointCloud);

    QAction *saveProj = fileMenu->addAction(tr("保存工程(&S)..."));
    saveProj->setShortcut(QKeySequence("Ctrl+S"));
    connect(saveProj, &QAction::triggered, this, &MainWindow::onSaveProjectFile);

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
}

void MainWindow::onStartReconstruction()
{
    if (m_pipelineRunner && m_pipelineRunner->isRunning())
        return;

    onStopReconstruction();
    delete m_pipelineRunner;
    m_pipelineRunner = nullptr;
    delete m_pipeline;
    m_pipeline = nullptr;

    m_pipeline = new ReconstructionPipeline();   // 注：非 QObject，无父对象
    m_pipeline->setConfig(m_configPanel->gatherConfig());

    m_pipelineRunner = new PipelineRunner(m_pipeline, this);

    const QString projectPath = m_configPanel->projectPath();
    const QString saveDir = !projectPath.isEmpty()
                                ? projectPath
                                : QCoreApplication::applicationDirPath() + "/output";
    m_pipelineRunner->setAutoSaveDirectory(saveDir);

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

void MainWindow::onSaveProjectFile()
{
    const QString filePath = QFileDialog::getSaveFileName(
        this, tr("保存工程"), QString(), tr("SFM 工程文件 (*.json);;所有文件 (*)"));
    if (filePath.isEmpty())
        return;
    if (m_pipeline) {
        m_pipeline->saveProject(filePath);
        m_logConsole->appendInfo(tr("工程已保存: %1").arg(filePath));
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
    if (success)
        loadAndDisplayCloudFromOutput();
}

void MainWindow::onStagePointCloudReady(const QVector<float> &points,
                                        const QVector<float> &colors,
                                        const QString &stageName)
{
    m_logConsole->appendInfo(tr("[%1] 阶段点云: %2 点")
                                 .arg(stageName).arg(points.size() / 3));
    m_pointCloudWidget->updatePointCloud(points, colors, true);
}

void MainWindow::onCameraPosesReady(const QVector<float> &poses)
{
    // PipelineRunner 当前未发射 cameraPosesReady；
    // 相机轨迹已通过 loadAndDisplayCloudFromOutput 读取 lastOutput 绘制。
    Q_UNUSED(poses);
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

    // 相机轨迹：中心点 + 朝向线段
    QVector<float> camCenters, camOrient;
    for (const CameraPose &pose : output.cameraPoses) {
        if (!pose.registered)
            continue;
        // 世界→相机: P_cam = R*P_world + t ⇒ 相机中心 = -R^T * t
        const cv::Mat t = pose.t;
        cv::Mat C = -pose.R.t() * t;
        camCenters.append(static_cast<float>(C.at<double>(0)));
        camCenters.append(static_cast<float>(C.at<double>(1)));
        camCenters.append(static_cast<float>(C.at<double>(2)));
        // 朝向（光轴在相机系为 (0,0,1)，转到世界系 = R^T * (0,0,1) = R 第三行）
        camOrient.append(static_cast<float>(pose.R.at<double>(2, 0)));
        camOrient.append(static_cast<float>(pose.R.at<double>(2, 1)));
        camOrient.append(static_cast<float>(pose.R.at<double>(2, 2)));
    }
    m_pointCloudWidget->setCameraPoses(camCenters, camOrient);
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