/**
 * @file ConfigPanel.cpp
 * @brief 左侧信息面板实现（项目摘要 + 结果区 + 显示选项）
 */

#include "ConfigPanel.h"

#include <QCheckBox>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QGroupBox>
#include <QLabel>
#include <QRegularExpression>
#include <QTextStream>
#include <QVBoxLayout>

#include "pipeline/ReconstructionConfigIO.h"
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

    // ── 显示选项 ──
    auto *displayBox = new QGroupBox(tr("显示"), container);
    auto *displayLayout = new QVBoxLayout(displayBox);
    m_cameraCheck = new QCheckBox(tr("相机"), displayBox);
    m_cameraCheck->setChecked(true);
    connect(m_cameraCheck, &QCheckBox::toggled,
            this, &ConfigPanel::cameraFrustaVisibleChanged);
    displayLayout->addWidget(m_cameraCheck);

    // 稀疏/密集点云显隐：二选一显示，但不用 radio 语义——
    // 勾选一个时由 MainWindow 接收信号后关闭另一个
    m_sparseCloudCheck = new QCheckBox(tr("稀疏点云"), displayBox);
    m_sparseCloudCheck->setEnabled(false);   // 重建/加载后按存在性启用
    connect(m_sparseCloudCheck, &QCheckBox::toggled,
            this, &ConfigPanel::sparseCloudVisibleChanged);
    displayLayout->addWidget(m_sparseCloudCheck);

    m_denseCloudCheck = new QCheckBox(tr("密集点云"), displayBox);
    m_denseCloudCheck->setEnabled(false);
    connect(m_denseCloudCheck, &QCheckBox::toggled,
            this, &ConfigPanel::denseCloudVisibleChanged);
    displayLayout->addWidget(m_denseCloudCheck);

    layout->addWidget(displayBox);

    layout->addStretch();

    setWidget(container);
}

ReconstructionConfig ConfigPanel::gatherConfig() const
{
    ReconstructionConfig config = sfm::makeDefaultReconstructionConfig();
    config.imageFiles = m_imageFiles;
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
}

void ConfigPanel::setFixedCameraMatrix(const cv::Mat &K, bool fixed)
{
    m_cameraMatrix = K.clone();
    m_fixCameraMatrix = fixed && !K.empty();
}

void ConfigPanel::setCameraFrustaChecked(bool checked)
{
    if (m_cameraCheck->isChecked() != checked) {
        QSignalBlocker blocker(m_cameraCheck);
        m_cameraCheck->setChecked(checked);
    }
}

void ConfigPanel::setCloudAvailability(bool sparse, bool dense)
{
    auto sync = [](QCheckBox *box, bool available) {
        box->setEnabled(available);
        if (!available && box->isChecked()) {
            QSignalBlocker blocker(box);
            box->setChecked(false);
        }
    };
    sync(m_sparseCloudCheck, sparse);
    sync(m_denseCloudCheck, dense);
}

void ConfigPanel::setSparseCloudChecked(bool checked)
{
    if (m_sparseCloudCheck->isChecked() != checked) {
        QSignalBlocker blocker(m_sparseCloudCheck);
        m_sparseCloudCheck->setChecked(checked);
    }
}

void ConfigPanel::setDenseCloudChecked(bool checked)
{
    if (m_denseCloudCheck->isChecked() != checked) {
        QSignalBlocker blocker(m_denseCloudCheck);
        m_denseCloudCheck->setChecked(checked);
    }
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
