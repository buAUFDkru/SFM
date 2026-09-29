/**
 * @file NewProjectDialog.cpp
 * @brief 新建项目对话框实现
 */

#include "NewProjectDialog.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QFileDialog>
#include <QFormLayout>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QRegularExpression>
#include <QPushButton>
#include <QStandardPaths>
#include <QTextStream>
#include <QVBoxLayout>

namespace
{
const QStringList kImageFilters = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"};

bool isValidProjectName(const QString &name)
{
    static const QRegularExpression illegal(R"([\\/:*?\"<>|])");
    return !name.trimmed().isEmpty() && !name.contains(illegal) &&
           !name.startsWith('.') && !name.endsWith('.');
}
} // namespace

NewProjectDialog::NewProjectDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("新建项目"));
    setMinimumWidth(460);

    auto *form = new QFormLayout;

    m_nameEdit = new QLineEdit(this);
    m_nameEdit->setPlaceholderText(tr("项目名称"));
    m_nameEdit->setClearButtonEnabled(true);
    form->addRow(tr("项目名称:"), m_nameEdit);

    m_locationEdit = new QLineEdit(this);
    m_locationEdit->setText(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    auto *locBtn = new QPushButton(tr("浏览..."), this);
    auto *locRow = new QHBoxLayout;
    locRow->addWidget(m_locationEdit, 1);
    locRow->addWidget(locBtn);
    form->addRow(tr("项目位置:"), locRow);

    m_imagesEdit = new QLineEdit(this);
    m_imagesEdit->setReadOnly(true);
    m_imagesEdit->setPlaceholderText(tr("选择图像目录或多张图像"));
    auto *dirBtn = new QPushButton(tr("选择目录"), this);
    auto *filesBtn = new QPushButton(tr("选择图像..."), this);
    auto *imgRow = new QHBoxLayout;
    imgRow->addWidget(m_imagesEdit, 1);
    imgRow->addWidget(dirBtn);
    imgRow->addWidget(filesBtn);
    form->addRow(tr("输入图像:"), imgRow);

    m_hintLabel = new QLabel(this);
    m_hintLabel->setStyleSheet("color: #888;");
    m_hintLabel->setWordWrap(true);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)->setText(tr("创建"));
    connect(buttons, &QDialogButtonBox::accepted, this, &NewProjectDialog::onAccept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    auto *layout = new QVBoxLayout(this);
    layout->addLayout(form);
    layout->addWidget(m_hintLabel);
    layout->addWidget(buttons);

    connect(m_nameEdit, &QLineEdit::textChanged, this, &NewProjectDialog::updateHint);
    connect(locBtn, &QPushButton::clicked, this, &NewProjectDialog::browseLocation);
    connect(dirBtn, &QPushButton::clicked, this, &NewProjectDialog::browseImageDirectory);
    connect(filesBtn, &QPushButton::clicked, this, &NewProjectDialog::browseImageFiles);

    updateHint();
}

QString NewProjectDialog::projectName() const
{
    return m_nameEdit->text().trimmed();
}

QString NewProjectDialog::projectLocation() const
{
    return m_locationEdit->text().trimmed();
}

QString NewProjectDialog::projectDir() const
{
    return QDir(projectLocation()).filePath(projectName());
}

QStringList NewProjectDialog::imageFiles() const
{
    return m_imageFiles;
}

void NewProjectDialog::browseLocation()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this, tr("选择项目位置"), m_locationEdit->text());
    if (!dir.isEmpty())
        m_locationEdit->setText(dir);
}

void NewProjectDialog::browseImageDirectory()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this, tr("选择图像目录"), QFileInfo(m_imagesEdit->text()).absolutePath());
    if (dir.isEmpty())
        return;

    QStringList files;
    QDirIterator it(dir, kImageFilters, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext())
        files.append(it.next());
    files.sort();
    setImageFiles(files);
}

void NewProjectDialog::browseImageFiles()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, tr("选择图像"),
        QFileInfo(m_imagesEdit->text()).absolutePath(),
        tr("图像 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"));
    if (!files.isEmpty())
        setImageFiles(files);
}

void NewProjectDialog::setImageFiles(const QStringList &files)
{
    m_imageFiles = files;
    m_cameraMatrix = cv::Mat();
    m_fixCameraMatrix = false;

    if (files.isEmpty()) {
        m_imagesEdit->clear();
        updateHint();
        return;
    }
    m_imagesEdit->setText(tr("%1 张图像").arg(files.size()));
    updateHint();
}

void NewProjectDialog::updateHint()
{
    QStringList hints;

    if (!m_nameEdit->text().trimmed().isEmpty() && !isValidProjectName(m_nameEdit->text()))
        hints << tr("项目名称包含非法字符");

    if (!m_imageFiles.isEmpty()) {
        // 内参检测：图像目录下存在 cameramatrix.txt 时将固定内参
        const QString kPath = QDir(QFileInfo(m_imageFiles.first()).absolutePath())
                                  .filePath(QStringLiteral("cameramatrix.txt"));
        if (QFile::exists(kPath)) {
            // 尝试解析（3×3 数字矩阵），成功才提示
            QFile f(kPath);
            bool ok = false;
            if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QTextStream in(&f);
                ok = true;
                for (int row = 0; row < 3 && ok; ++row) {
                    const QString line = in.readLine();
                    const QStringList parts =
                        line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
                    if (parts.size() < 3) { ok = false; break; }
                    for (int col = 0; col < 3; ++col) {
                        bool numOk = false;
                        parts[col].toDouble(&numOk);
                        if (!numOk) { ok = false; break; }
                    }
                }
            }
            if (ok) {
                m_fixCameraMatrix = true;
                hints << tr("检测到内参文件 cameramatrix.txt，重建时将固定内参");
            }
        }
    }

    m_hintLabel->setText(hints.join(QStringLiteral("；")));
}

void NewProjectDialog::onAccept()
{
    if (!isValidProjectName(projectName())) {
        QMessageBox::warning(this, tr("新建项目"), tr("请输入合法的项目名称。"));
        return;
    }

    const QFileInfo loc(projectLocation());
    if (!loc.isDir()) {
        QMessageBox::warning(this, tr("新建项目"), tr("项目位置不存在，请重新选择。"));
        return;
    }

    QDir dir(projectDir());
    if (dir.exists() && !dir.isEmpty()) {
        QMessageBox::warning(this, tr("新建项目"),
                             tr("目标目录已存在且非空：\n%1\n请更换项目名称或位置。")
                                 .arg(projectDir()));
        return;
    }

    if (m_imageFiles.isEmpty()) {
        QMessageBox::warning(this, tr("新建项目"), tr("请先选择输入图像。"));
        return;
    }

    accept();
}
