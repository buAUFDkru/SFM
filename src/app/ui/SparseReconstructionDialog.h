#ifndef SPARSERECONSTRUCTIONDIALOG_H
#define SPARSERECONSTRUCTIONDIALOG_H

#include <QDialog>
#include <QJsonObject>

class AlgorithmComboPanel;
class ControlButtons;
class PluginRegistry;

/**
 * @brief 稀疏重建配置对话框（非模态，经菜单"重建 → 稀疏重建"打开）
 *
 * 承载原左侧面板"重建任务区"的内容：特征描述符/特征匹配器算法选择
 * （schema 驱动参数表单）+ 开始/停止按钮。对话框隐藏仅收起 UI，
 * 实例随主窗口存续，重建运行中可随时重新打开查看/停止。
 *
 * 算法参数由本对话框收集（gatherAlgorithms），图像列表/项目路径/
 * 固定内参仍由 ConfigPanel 持有，MainWindow 负责合并为 ReconstructionConfig。
 */
class SparseReconstructionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SparseReconstructionDialog(PluginRegistry *registry,
                                        QWidget *parent = nullptr);

    /// 收集当前算法选择与参数（写入调用方的配置字段）
    void gatherAlgorithms(QString &descriptorType, QJsonObject &descriptorParams,
                          QString &matcherType, QJsonObject &matcherParams) const;

    /// 回填算法配置（打开项目恢复清单时；不触发 configChanged）
    void setAlgorithms(const QString &descriptorType, const QJsonObject &descriptorParams,
                       const QString &matcherType, const QJsonObject &matcherParams);

    /// 重建运行状态：运行中锁定算法配置，停止按钮可用
    void setRunning(bool running);

    /// 项目可用性门控（未加载项目时禁用开始按钮）
    void setStartEnabled(bool enabled);

signals:
    void startRequested();
    void stopRequested();
    void configChanged();

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void setupUI();

    PluginRegistry *m_registry = nullptr;
    AlgorithmComboPanel *m_descriptor = nullptr;
    AlgorithmComboPanel *m_matcher = nullptr;
    ControlButtons *m_controls = nullptr;
    bool m_updating = false;  ///< 回填算法配置期间抑制 configChanged
};

#endif // SPARSERECONSTRUCTIONDIALOG_H
