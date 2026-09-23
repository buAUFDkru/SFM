#ifndef CONFIGPANEL_H
#define CONFIGPANEL_H

#include <QScrollArea>
#include <QStringList>

#include "pipeline/ReconstructionPipeline.h"

class AlgorithmComboPanel;
class ControlButtons;
class PluginRegistry;
class QLineEdit;
class QLabel;

/**
 * @brief 左侧参数配置面板（schema 驱动）
 *
 * 组织"项目/图像输入 + 描述符算法 + 匹配器算法 + 控制按钮"。
 * gatherConfig() 将 UI 状态打包为 ReconstructionConfig
 * （其中 descriptorParams / matcherParams 由 AlgorithmComboPanel 按插件 schema 收集）。
 */
class ConfigPanel : public QScrollArea
{
    Q_OBJECT

public:
    ConfigPanel(PluginRegistry *registry, QWidget *parent = nullptr);

    ReconstructionConfig gatherConfig() const;

    void setImageFiles(const QStringList &files);
    void setImageDirectory(const QString &dir);
    int imageCount() const;

    QString projectPath() const;
    void setProjectPath(const QString &path);

signals:
    void startRequested();
    void stopRequested();
    void configChanged();

private:
    void setupUI();
    void tryLoadCameraMatrix(const QString &dir);

    PluginRegistry *m_registry;
    AlgorithmComboPanel *m_descriptor = nullptr;
    AlgorithmComboPanel *m_matcher = nullptr;
    ControlButtons *m_controls = nullptr;

    QLineEdit *m_projectEdit = nullptr;
    QLineEdit *m_imagesEdit = nullptr;
    QLabel *m_importHint = nullptr;

    QStringList m_imageFiles;
    QString m_projectPath;
    cv::Mat m_cameraMatrix;
    bool m_fixCameraMatrix = false;
};

#endif // CONFIGPANEL_H