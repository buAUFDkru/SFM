#ifndef ALGORITHMCOMBOPANEL_H
#define ALGORITHMCOMBOPANEL_H

#include <QJsonObject>
#include <QWidget>

#include "plugin/ParamSchema.h"

class QComboBox;
class ParamFormGenerator;
class PluginRegistry;

/**
 * @brief 算法选择面板（schema 驱动）
 *
 * 组合"算法下拉框（来自 PluginRegistry::byCategory）+ 参数表单"。
 * 切换算法时按新插件的 paramSchema() 重建 ParamFormGenerator。
 *
 * 对外提供：
 *   - currentAlgorithm() 当前选中算法 id
 *   - collectParams()    { algorithm: id, params: {...} }（params 即 PluginTypes 参数）
 */
class AlgorithmComboPanel : public QWidget
{
    Q_OBJECT

public:
    /// category: "descriptor" | "matcher"（用于 byCategory 过滤）
    AlgorithmComboPanel(const QString &category,
                        PluginRegistry *registry,
                        const QString &title,
                        QWidget *parent = nullptr);

    /// 当前选中的算法 id（未加载任何算法时返回空串）
    QString currentAlgorithm() const;

    /// 收集算法 id 与参数
    QJsonObject collectParams() const;

    /// 回填参数（算法 id 存在时切换并回填；不存在仅回填表单）
    void setParams(const QString &algorithm, const QJsonObject &params);

    /// 面板标题
    QString category() const { return m_category; }

signals:
    /// 当前算法或参数发生变化
    void changed();

private:
    void rebuildForm();

    QString m_category;
    PluginRegistry *m_registry;
    QString m_title;

    QComboBox *m_algoCombo = nullptr;
    ParamFormGenerator *m_form = nullptr;
};

#endif // ALGORITHMCOMBOPANEL_H