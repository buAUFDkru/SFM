/**
 * @file ConfigPanel.cpp
 * @brief 左侧参数配置面板实现（schema 驱动）
 */

#include "ConfigPanel.h"
#include "ControlButtons.h"
#include "ParamWidget/AlgorithmComboPanel.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGroupBox>
#include <QLabel>
#include <QRegularExpression>
#include <QTextStream>
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

    // ── 描述符算法面板 ──
    m_descriptor = new AlgorithmComboPanel(QStringLiteral("descriptor"), m_registry,
                                           tr("特征描述符"), container);
    layout->addWidget(m_descriptor);

    QFrame *line1 = new QFrame(container);
    line1->setFrameShape(QFrame::HLine);
    line1->setFrameShadow(QFrame::Sunken);
    layout->addWidget(line1);

    // ── 匹配器算法面板 ──
    m_matcher = new AlgorithmComboPanel(QStringLiteral("matcher"), m_registry,
                                        tr("特征匹配器"), container);
    layout->addWidget(m_matcher);

    // 默认算法选择：描述符 SIFT、匹配器 FLANN。
    // 下拉框填充顺序来自注册表（QHash 无序），必须显式指定默认项。
    // 参数传空对象 → 表单按各插件 paramSchema 的默认值填充。
    m_descriptor->setParams(QStringLiteral("SIFT"), QJsonObject{});
    m_matcher->setParams(QStringLiteral("FLANN"), QJsonObject{});

    QFrame *line2 = new QFrame(container);
    line2->setFrameShape(QFrame::HLine);
    line2->setFrameShadow(QFrame::Sunken);
    layout->addWidget(line2);

    // ── 控制按钮 ──
    m_controls = new ControlButtons(container);
    layout->addWidget(m_controls);

    layout->addStretch();

    setWidget(container);

    connect(m_controls, &ControlButtons::startClicked, this, &ConfigPanel::startRequested);
    connect(m_controls, &ControlButtons::stopClicked, this, &ConfigPanel::stopRequested);
    connect(m_descriptor, &AlgorithmComboPanel::changed, this, &ConfigPanel::configChanged);
    connect(m_matcher, &AlgorithmComboPanel::changed, this, &ConfigPanel::configChanged);

    // 未加载项目时禁用开始按钮
    setProjectLoaded(false);
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
