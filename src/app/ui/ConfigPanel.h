#ifndef CONFIGPANEL_H
#define CONFIGPANEL_H

#include <QScrollArea>
#include <QStringList>

#include "pipeline/ReconstructionPipeline.h"

class PluginRegistry;
class QCheckBox;
class QLabel;

/**
 * @brief 左侧信息面板（上下文化两段式）
 *
 * 自上而下：
 *   1. 项目摘要（创建/打开由菜单触发）
 *   2. 显示选项：「相机」复选框控制相机位姿金字塔显隐（默认开）
 *
 * 算法选择与开始/停止按钮已移至"重建 → 稀疏重建"对话框
 * （SparseReconstructionDialog）；密集重建入口仅保留菜单
 * "重建 → 密集重建"。本面板 gatherConfig() 只含图像列表与
 * 固定内参，算法字段由对话框收集、MainWindow 合并。
 */
class ConfigPanel : public QScrollArea
{
    Q_OBJECT

public:
    ConfigPanel(PluginRegistry *registry, QWidget *parent = nullptr);

    /// 基础配置：imageFiles + cameraMatrix/fixCameraMatrix（算法字段由对话框收集）
    ReconstructionConfig gatherConfig() const;

    void setImageFiles(const QStringList &files);
    void setImageDirectory(const QString &dir);
    int imageCount() const;

    /// 显示项目摘要（无项目时 name 传空串显示占位提示）
    void setProjectSummary(const QString &name, const QString &detail);

    /// 记录项目加载状态（影响固定内参覆盖保护；开始按钮由对话框控制）
    void setProjectLoaded(bool loaded);

    /// 设置固定内参（打开项目时按清单恢复；空 Mat 表示清除）
    void setFixedCameraMatrix(const cv::Mat &K, bool fixed);

    /// 程序化同步「相机」复选框（视图菜单切换时；不触发信号）
    void setCameraFrustaChecked(bool checked);

signals:
    void configChanged();
    /// 「相机」复选框切换（金字塔显隐）
    void cameraFrustaVisibleChanged(bool visible);

private:
    void setupUI();
    void tryLoadCameraMatrix(const QString &dir);

    PluginRegistry *m_registry;

    // 项目摘要
    QLabel *m_projectLabel = nullptr;
    QLabel *m_projectDetail = nullptr;
    QLabel *m_importHint = nullptr;

    // 显示选项
    QCheckBox *m_cameraCheck = nullptr;

    QStringList m_imageFiles;
    cv::Mat m_cameraMatrix;
    bool m_fixCameraMatrix = false;
    bool m_projectLoaded = false;
};

#endif // CONFIGPANEL_H
