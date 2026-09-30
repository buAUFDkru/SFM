/**
 * @file ConfigPanel.cpp
 * @brief 左侧参数配置面板实现（schema 驱动，上下文化三段式）
 */

#include "ConfigPanel.h"
#include "ControlButtons.h"
#include "ParamWidget/AlgorithmComboPanel.h"

#include <QCheckBox>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QRegularExpression>
#include <QTextStream>
#include <QToolButton>
#include <QVBoxLayout>

#include "plugin/PluginRegistry.h"

ConfigPanel::ConfigPanel(PluginRegistry *registry, QWidget *parent)
    : QScrollArea(parent)
    , m_registry(registry)
{
    setWidgetResizable(true);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setupUI();
}

void ConfigPanel::setupUI()
{
    QWidget *container = new QWidget(this);
    container->setObjectName("configContainer");
    QVBoxLayout *layout = new QVBoxLayout(container);
    layout->setSpacing(8);
    layout->setContentsMargins(8, 8, 8, 8);

    // ── 项目摘要（创建/打开由菜单驱动，这里仅展示） ──
    auto *projectBox = new QGroupBox(tr("项目"), container);
    auto *projectLayout = new QVBoxLayout(projectBox);

    m_projectLabel = new QLabel(tr("未打开项目"), projectBox);
    m_projectLabel->setObjectName("projectLabel");
    m_projectLabel->setStyleSheet("font-weight: bold;");
    m_projectLabel->setWordWrap(true);
    projectLayout->addWidget(m_projectLabel);

    m_projectDetail = new QLabel(tr("通过「文件 → 新建项目」开始"), projectBox);
    m_projectDetail->setStyleSheet("color: #888;");
    m_projectDetail->setWordWrap(true);
    projectLayout->addWidget(m_projectDetail);

    m_importHint = new QLabel(projectBox);
    m_importHint->setStyleSheet("color: #888;");
    m_importHint->setWordWrap(true);
    projectLayout->addWidget(m_importHint);

    layout->addWidget(projectBox);

    // ── 结果区（稀疏完成后显示） ──
    m_resultBox = new QGroupBox(tr("结果"), container);
    m_resultBox->setObjectName(QStringLiteral("resultBox"));
    auto *resultLayout = new QVBoxLayout(m_resultBox);

    m_sparseResultLabel = new QLabel(m_resultBox);
    m_sparseResultLabel->setWordWrap(true);
    resultLayout->addWidget(m_sparseResultLabel);

    m_denseResultLabel = new QLabel(m_resultBox);
    m_denseResultLabel->setStyleSheet("color: #888;");
    m_denseResultLabel->setWordWrap(true);
    m_denseResultLabel->setVisible(false);
    resultLayout->addWidget(m_denseResultLabel);

    m_denseButton = new QPushButton(tr("密集重建..."), m_resultBox);
    connect(m_denseButton, &QPushButton::clicked,
            this, &ConfigPanel::denseReconstructionRequested);
    resultLayout->addWidget(m_denseButton);

    m_resultBox->setVisible(false);
    layout->addWidget(m_resultBox);

    // ── 重建任务区（可折叠：标题按钮 + 内容容器） ──
    m_taskToggle = new QToolButton(container);
    m_taskToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_taskToggle->setAutoRaise(true);
    m_taskToggle->setText(tr("重建任务"));
    connect(m_taskToggle, &QToolButton::clicked,
            this, &ConfigPanel::toggleTaskSection);
    layout->addWidget(m_taskToggle);

    m_taskBody = new QWidget(container);
    m_taskBody->setObjectName(QStringLiteral("taskBody"));
    auto *taskLayout = new QVBoxLayout(m_taskBody);
    taskLayout->setContentsMargins(0, 0, 0, 0);
    taskLayout->setSpacing(8);

    m_descriptor = new AlgorithmComboPanel(QStringLiteral("descriptor"), m_registry,
                                           tr("特征描述符"), m_taskBody);
    taskLayout->addWidget(m_descriptor);

    QFrame *line1 = new QFrame(m_taskBody);
    line1->setFrameShape(QFrame::HLine);
    line1->setFrameShadow(QFrame::Sunken);
    taskLayout->addWidget(line1);

    m_matcher = new AlgorithmComboPanel(QStringLiteral("matcher"), m_registry,
                                        tr("特征匹配器"), m_taskBody);
    taskLayout->addWidget(m_matcher);

    QFrame *line2 = new QFrame(m_taskBody);
    line2->setFrameShape(QFrame::HLine);
    line2->setFrameShadow(QFrame::Sunken);
    taskLayout->addWidget(line2);

    m_controls = new ControlButtons(m_taskBody);
    taskLayout->addWidget(m_controls);

    layout->addWidget(m_taskBody);

    // ── 显示选项 ──
    auto *displayBox = new QGroupBox(tr("显示"), container);
    auto *displayLayout = new QVBoxLayout(displayBox);
    m_cameraCheck = new QCheckBox(tr("相机位姿金字塔"), displayBox);
    m_cameraCheck->setChecked(true);
    connect(m_cameraCheck, &QCheckBox::toggled,
            this, &ConfigPanel::cameraFrustaVisibleChanged);
    displayLayout->addWidget(m_cameraCheck);
    layout->addWidget(displayBox);

    // 默认算法选择：描述符 SIFT、匹配器 FLANN。
    // 下拉框填充顺序来自注册表（QHash 无序），必须显式指定默认项。
    // 参数传空对象 → 表单按各插件 paramSchema 的默认值填充。
    m_descriptor->setParams(QStringLiteral("SIFT"), QJsonObject{});
    m_matcher->setParams(QStringLiteral("FLANN"), QJsonObject{});

    layout->addStretch();

    setWidget(container);

    connect(m_controls, &ControlButtons::startClicked, this, &ConfigPanel::startRequested);
    connect(m_controls, &ControlButtons::stopClicked, this, &ConfigPanel::stopRequested);
    connect(m_descriptor, &AlgorithmComboPanel::changed, this, &ConfigPanel::configChanged);
    connect(m_matcher, &AlgorithmComboPanel::changed, this, &ConfigPanel::configChanged);

    // 未加载项目时禁用开始按钮
    setProjectLoaded(false);
    setTaskSectionExpanded(true);
}

ReconstructionConfig ConfigPanel::gatherConfig() const
{
    ReconstructionConfig config;
    config.imageFiles = m_imageFiles;

    const QJsonObject desc = m_descriptor->collectParams();
    config.descriptorType = desc.value(QStringLiteral("algorithm")).toString();
    config.descriptorParams = desc.value(QStringLiteral("params")).toObject();

    const QJsonObject matcher = m_matcher->collectParams();
    config.matcherType = matcher.value(QStringLiteral("algorithm")).toString();
    config.matcherParams = matcher.value(QStringLiteral("params")).toObject();

    config.cameraMatrix = m_cameraMatrix;
    config.fixCameraMatrix = m_fixCameraMatrix;

    return config;
}

void ConfigPanel::setProjectSummary(const QString &name, const QString &detail)
{
    m_projectLabel->setText(name.isEmpty() ? tr("未打开项目") : name);
    if (!detail.isEmpty())
        m_projectDetail->setText(detail);
    else if (name.isEmpty())
        m_projectDetail->setText(tr("通过「文件 → 新建项目」开始"));
}

void ConfigPanel::setProjectLoaded(bool loaded)
{
    m_projectLoaded = loaded;
    m_controls->setStartEnabled(loaded);
}

void ConfigPanel::setRunning(bool running)
{
    m_running = running;
    // 运行中锁定算法配置；按钮内部已有运行态门控
    m_descriptor->setEnabled(!running);
    m_matcher->setEnabled(!running);
    m_controls->setRunning(running);
    if (running)
        setTaskSectionExpanded(true);   // 运行中必须可见（停止按钮）
}

void ConfigPanel::setSparseResult(int points)
{
    m_sparseResultLabel->setText(tr("稀疏完成: %1 点").arg(points));
    m_resultBox->setVisible(true);
    setTaskSectionExpanded(false);   // 完成后自动折叠任务区
}

void ConfigPanel::setDenseResult(int points)
{
    m_denseResultLabel->setText(points >= 0
                                    ? tr("密集完成: %1 点").arg(points)
                                    : tr("密集: 已有结果"));
    m_denseResultLabel->setVisible(true);
    m_resultBox->setVisible(true);
}

void ConfigPanel::resetResults()
{
    m_resultBox->setVisible(false);
    m_sparseResultLabel->clear();
    m_denseResultLabel->setVisible(false);
    m_denseResultLabel->clear();
    setTaskSectionExpanded(true);
}

void ConfigPanel::setCameraFrustaChecked(bool checked)
{
    if (m_cameraCheck->isChecked() != checked) {
        QSignalBlocker blocker(m_cameraCheck);
        m_cameraCheck->setChecked(checked);
    }
}

void ConfigPanel::toggleTaskSection()
{
    setTaskSectionExpanded(!m_taskExpanded);
}

void ConfigPanel::setTaskSectionExpanded(bool expanded)
{
    m_taskExpanded = expanded;
    m_taskBody->setVisible(expanded);
    m_taskToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    if (expanded)
        m_taskToggle->setText(tr("重建任务"));
    else
        m_taskToggle->setText(tr("重建任务（已完成，点击展开）"));
}

void ConfigPanel::setAlgorithms(const QString &descriptorType, const QJsonObject &descriptorParams,
                                const QString &matcherType, const QJsonObject &matcherParams)
{
    m_updating = true;
    m_descriptor->setParams(descriptorType, descriptorParams);
    m_matcher->setParams(matcherType, matcherParams);
    m_updating = false;
}

void ConfigPanel::setFixedCameraMatrix(const cv::Mat &K, bool fixed)
{
    m_cameraMatrix = K.clone();
    m_fixCameraMatrix = fixed && !K.empty();
}

void ConfigPanel::setImageFiles(const QStringList &files)
{
    m_imageFiles = files;
    if (files.isEmpty()) {
        m_importHint->clear();
        if (!m_projectLoaded)
            m_projectDetail->setText(tr("通过「文件 → 新建项目」开始"));
        else
            m_projectDetail->setText(tr("0 张图像"));
        return;
    }
    m_projectDetail->setText(tr("%1 张图像").arg(files.size()));
    m_importHint->setText(QStringLiteral("%1 ... %2")
                              .arg(QFileInfo(files.first()).fileName())
                              .arg(QFileInfo(files.last()).fileName()));
    tryLoadCameraMatrix(QFileInfo(files.first()).absolutePath());
    if (!m_updating)
        emit configChanged();
}

void ConfigPanel::setImageDirectory(const QString &dir)
{
    QStringList filters = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"};
    QStringList files;
    QDirIterator it(dir, filters, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext())
        files.append(it.next());
    files.sort();
    setImageFiles(files);
}

void ConfigPanel::tryLoadCameraMatrix(const QString &dir)
{
    // 项目模式下固定内参由清单管理，覆盖图像目录中的同名文件
    if (m_projectLoaded && m_fixCameraMatrix)
        return;

    m_cameraMatrix = cv::Mat();
    m_fixCameraMatrix = false;

    const QString filePath = QDir(dir).filePath("cameramatrix.txt");
    if (!QFile::exists(filePath))
        return;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    QTextStream in(&file);
    for (int row = 0; row < 3; ++row) {
        const QString line = in.readLine();
        if (line.isNull())
            return;
        const QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.size() < 3)
            return;
        for (int col = 0; col < 3; ++col) {
            bool ok = false;
            K.at<double>(row, col) = parts[col].toDouble(&ok);
            if (!ok)
                return;
        }
    }

    m_cameraMatrix = K;
    m_fixCameraMatrix = true;
}

int ConfigPanel::imageCount() const
{
    return m_imageFiles.size();
}
