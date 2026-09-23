#ifndef PARAMFORMGENERATOR_H
#define PARAMFORMGENERATOR_H

#include <QJsonObject>
#include <QWidget>

#include "plugin/ParamSchema.h"

/**
 * @brief schema 驱动参数表单生成器
 *
 * 输入 QList<ParamSchema>（插件 paramSchema() 的产物），按 group 分组生成表单控件：
 *   - Int    → QSpinBox
 *   - Double → QDoubleSpinBox
 *   - Bool   → QCheckBox
 *   - String → QLineEdit
 *   - Enum   → QComboBox（enumValues）
 *   - Path   → QLineEdit + 浏览按钮
 *
 * 用户修改任意控件即通过 valuesChanged 通知外部；collectParams() 导出
 * QJsonObject，与 ReconstructionConfig 的 descriptorParams/matcherParams 直接打通。
 */
class ParamFormGenerator : public QWidget
{
    Q_OBJECT

public:
    explicit ParamFormGenerator(QWidget *parent = nullptr);

    /// 设置参数协议并重建全部控件（按 group 分组）
    void setSchemas(const QList<ParamSchema> &schemas);

    /// 将当前控件值导出为 QJsonObject；未列出的 schema 使用默认值
    QJsonObject collectParams() const;

    /// 用已有参数回填控件；不存在的 key 还原为 schema 默认值
    void setParams(const QJsonObject &params);

    /// 当前协议是否为空
    bool isEmpty() const { return m_schemas.isEmpty(); }

signals:
    /// 任一参数控件被用户修改后发出
    void valuesChanged();

private:
    /// 依据 schema 创建对应编辑控件
    QWidget *createEditor(const ParamSchema &schema);

    QList<ParamSchema> m_schemas;
};

#endif // PARAMFORMGENERATOR_H