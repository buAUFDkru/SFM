#ifndef CONFIGPANEL_H
#define CONFIGPANEL_H

#include <QScrollArea>
#include <QStringList>

#include "pipeline/ReconstructionPipeline.h"

class AlgorithmComboPanel;
class ControlButtons;
class PluginRegistry;
class QLabel;

/**
 * @brief 左侧参数配置面板（schema 驱动）
 *
 * 组织"项目摘要 + 描述符算法 + 匹配器算法 + 控制按钮"。
 * gatherConfig() 将 UI 状态打包为 ReconstructionConfig
 * （其中 descriptorParams / matcherParams 由 AlgorithmComboPanel 按插件 schema 收集；
 *   项目路径与名称由 MainWindow 持有并回填到 config）。
 *
 * 项目管理（创建/打开）由菜单触发，本面板仅显示当前项目摘要；
 * 未加载项目时开始按钮禁用。
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

    /// 显示项目摘要（无项目时 name 传空串显示占位提示）
    void setProjectSummary(const QString &name, const QString &detail);

    /// 项目是否已加载：未加载时禁用开始按钮
    void setProjectLoaded(bool loaded);

    /// 恢复算法配置（打开项目时按清单回填下拉框与参数表单）
    void setAlgorithms(const QString &descriptorType, const QJsonObject &descriptorParams,
                       const QString &matcherType, const QJsonObject &matcherParams);

    /// 设置固定内参（打开项目时按清单恢复；空 Mat 表示清除）
    void setFixedCameraMatrix(const cv::Mat &K, bool fixed);

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

    QLabel *m_projectLabel = nullptr;
    QLabel *m_projectDetail = nullptr;
    QLabel *m_importHint = nullptr;

    QStringList m_imageFiles;
    cv::Mat m_cameraMatrix;
    bool m_fixCameraMatrix = false;
    bool m_projectLoaded = false;
    bool m_updating = false;  ///< 回填算法配置期间抑制 configChanged
};

#endif // CONFIGPANEL_H
