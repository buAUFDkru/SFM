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
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
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

    // ── 项目与图像输入 ──
    auto *inputBox = new QGroupBox(tr("输入"), container);
    auto *inputLayout = new QVBoxLayout(inputBox);

    auto *projRow = new QHBoxLayout();
    projRow->addWidget(new QLabel(tr("项目:"), inputBox));
    m_projectEdit = new QLineEdit(inputBox);
    m_projectEdit->setPlaceholderText(tr("输出目录"));
    auto *projBtn = new QToolButton(inputBox);
    projBtn->setText(QStringLiteral("..."));
    connect(projBtn, &QToolButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("选择项目输出目录"), m_projectEdit->text());
        if (!dir.isEmpty())
            setProjectPath(dir);
    });
    projRow->addWidget(m_projectEdit, 1);
    projRow->addWidget(projBtn);
    inputLayout->addLayout(projRow);

    auto *imgRow = new QHBoxLayout();
    imgRow->addWidget(new QLabel(tr("图像:"), inputBox));
    m_imagesEdit = new QLineEdit(inputBox);
    m_imagesEdit->setPlaceholderText(tr("图像文件或目录（可多选）"));
    m_imagesEdit->setReadOnly(true);
    auto *imgBtn = new QToolButton(inputBox);
    imgBtn->setText(QStringLiteral("..."));
    connect(imgBtn, &QToolButton::clicked, this, [this]() {
        const QStringList files = QFileDialog::getOpenFileNames(
            this, tr("选择图像"),
            QFileInfo(m_projectEdit->text()).absolutePath(),
            tr("图像 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"));
        if (!files.isEmpty())
            setImageFiles(files);
    });
    auto *dirBtn = new QToolButton(inputBox);
    dirBtn->setText(tr("目录"));
    connect(dirBtn, &QToolButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(this, tr("选择图像目录"));
        if (!dir.isEmpty())
            setImageDirectory(dir);
    });
    imgRow->addWidget(m_imagesEdit, 1);
    imgRow->addWidget(dirBtn);
    imgRow->addWidget(imgBtn);
    inputLayout->addLayout(imgRow);

    m_importHint = new QLabel(inputBox);
    m_importHint->setStyleSheet("color: #888;");
    m_importHint->setWordWrap(true);
    inputLayout->addWidget(m_importHint);

    layout->addWidget(inputBox);

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
}

ReconstructionConfig ConfigPanel::gatherConfig() const
{
    ReconstructionConfig config;
    config.imageFiles = m_imageFiles;
    config.projectPath = m_projectPath;

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

void ConfigPanel::setImageFiles(const QStringList &files)
{
    m_imageFiles = files;
    if (files.isEmpty()) {
        m_imagesEdit->clear();
        m_importHint->setText(QString());
        return;
    }
    m_imagesEdit->setText(QString::number(files.size()) + tr(" 张图像"));
    m_importHint->setText(QStringLiteral("%1 ... %2")
                              .arg(QFileInfo(files.first()).fileName())
                              .arg(QFileInfo(files.last()).fileName()));
    tryLoadCameraMatrix(QFileInfo(files.first()).absolutePath());
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

QString ConfigPanel::projectPath() const
{
    return m_projectPath;
}

void ConfigPanel::setProjectPath(const QString &path)
{
    m_projectPath = path;
    m_projectEdit->setText(path);
    emit configChanged();
}