#ifndef CONFIGPANEL_H
#define CONFIGPANEL_H

#include <QScrollArea>
#include <QStringList>

#include "pipeline/ReconstructionPipeline.h"

class AlgorithmComboPanel;
class ControlButtons;
class PluginRegistry;
class QCheckBox;
class QGroupBox;
class QLabel;
class QPushButton;
class QToolButton;
class QWidget;

/**
 * @brief 左侧参数配置面板（schema 驱动，上下文化三段式）
 *
 * 自上而下：
 *   1. 项目摘要（创建/打开由菜单触发）
 *   2. 结果区：稀疏完成后显示点数/误差/耗时 + "密集重建…"按钮（初始隐藏）
 *   3. 重建任务区（可折叠）：描述符/匹配器算法 + 开始/停止；
 *      稀疏完成后自动折叠，可手动展开重新调参重跑；运行中锁定配置
 *   4. 显示选项：「相机」复选框控制相机位姿金字塔显隐（默认开）
 *
 * gatherConfig() 将 UI 状态打包为 ReconstructionConfig
 * （descriptorParams / matcherParams 由 AlgorithmComboPanel 按插件 schema 收集；
 *   项目路径与名称由 MainWindow 持有并回填到 config）。
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

    /// 重建运行状态：运行中锁定算法配置并强制展开任务区
    void setRunning(bool running);

    /// 稀疏重建完成：显示结果区并自动折叠任务区
    void setSparseResult(int points);

    /// 密集重建完成：结果区追加密集行（points<0 表示仅有结果文件无点数）
    void setDenseResult(int points);

    /// 清空结果区并展开任务区（新建项目/重置时）
    void resetResults();

    /// 程序化同步「相机」复选框（视图菜单切换时；不触发信号）
    void setCameraFrustaChecked(bool checked);

signals:
    void startRequested();
    void stopRequested();
    void configChanged();
    /// 结果区"密集重建…"按钮
    void denseReconstructionRequested();
    /// 「相机」复选框切换（金字塔显隐）
    void cameraFrustaVisibleChanged(bool visible);

private slots:
    void toggleTaskSection();

private:
    void setupUI();
    void tryLoadCameraMatrix(const QString &dir);
    void setTaskSectionExpanded(bool expanded);

    PluginRegistry *m_registry;
    AlgorithmComboPanel *m_descriptor = nullptr;
    AlgorithmComboPanel *m_matcher = nullptr;
    ControlButtons *m_controls = nullptr;

    // 项目摘要
    QLabel *m_projectLabel = nullptr;
    QLabel *m_projectDetail = nullptr;
    QLabel *m_importHint = nullptr;

    // 结果区
    QGroupBox *m_resultBox = nullptr;
    QLabel *m_sparseResultLabel = nullptr;
    QLabel *m_denseResultLabel = nullptr;
    QPushButton *m_denseButton = nullptr;

    // 重建任务区（可折叠）
    QToolButton *m_taskToggle = nullptr;
    QWidget *m_taskBody = nullptr;

    // 显示选项
    QCheckBox *m_cameraCheck = nullptr;

    QStringList m_imageFiles;
    cv::Mat m_cameraMatrix;
    bool m_fixCameraMatrix = false;
    bool m_projectLoaded = false;
    bool m_running = false;
    bool m_taskExpanded = true;
    bool m_updating = false;  ///< 回填算法配置期间抑制 configChanged
};

#endif // CONFIGPANEL_H
