#ifndef NEWPROJECTDIALOG_H
#define NEWPROJECTDIALOG_H

#include <QDialog>

#include <QStringList>

#include "data/SfmTypes.h"

class QLineEdit;
class QLabel;

/**
 * @brief 新建项目对话框
 *
 * 收集项目名称、存放位置与输入图像（引用原路径，不复制），
 * 并检测图像目录中的 cameramatrix.txt 固定内参（仅提示，
 * 实际加载由 ConfigPanel 在设置图像列表时完成）。
 */
class NewProjectDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NewProjectDialog(QWidget *parent = nullptr);

    QString projectName() const;
    QString projectLocation() const;   ///< 项目父目录
    QString projectDir() const;        ///< projectLocation()/projectName()
    QStringList imageFiles() const;

private slots:
    void browseLocation();
    void browseImageDirectory();
    void browseImageFiles();
    void updateHint();
    void onAccept();

private:
    QLineEdit *m_nameEdit = nullptr;
    QLineEdit *m_locationEdit = nullptr;
    QLineEdit *m_imagesEdit = nullptr;
    QLabel *m_hintLabel = nullptr;

    QStringList m_imageFiles;
    cv::Mat m_cameraMatrix;      ///< 图像目录 cameramatrix.txt 检测结果（提示用）
    bool m_fixCameraMatrix = false;

    void setImageFiles(const QStringList &files);
};

#endif // NEWPROJECTDIALOG_H
