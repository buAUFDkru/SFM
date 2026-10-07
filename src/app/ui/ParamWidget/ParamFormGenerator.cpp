/**
 * @file ParamFormGenerator.cpp
 * @brief schema 驱动参数表单生成器实现
 *
 * 按 ParamSchema 协议自动生成分组表单控件，控件值与 QJsonObject
 * 双向转换。供 AlgorithmComboPanel / 密集重建对话框复用。
 */

#include "ParamFormGenerator.h"
#include "ParamWheelFilter.h"

#include <limits>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

ParamFormGenerator::ParamFormGenerator(QWidget *parent)
    : QWidget(parent)
{
}

void ParamFormGenerator::setSchemas(const QList<ParamSchema> &schemas)
{
    m_schemas = schemas;

    // 释放旧内容：从布局逐项取下并删除顶层控件（其子控件随级联销毁），
    // 再删除布局对象（含 stretch 空隙项）。
    // 注意不能用 qDeleteAll(findChildren<QWidget*>())：返回列表同时包含
    // 组框与其后代，先删组框会级联销毁子控件，循环再次 delete 同一指针
    // 造成双重释放（切换算法闪退的根因）。
    if (QLayout *oldLayout = layout()) {
        while (QLayoutItem *item = oldLayout->takeAt(0)) {
            delete item->widget();   // widget 为空时 delete 空指针安全（空隙项）
            delete item;
        }
        delete oldLayout;
    }

    if (m_schemas.isEmpty()) {
        auto *v = new QVBoxLayout(this);
        auto *hint = new QLabel(tr("该算法无配置参数"), this);
        hint->setStyleSheet("color: #888;");
        v->addWidget(hint);
        return;
    }

    // 按 group 分组（保持首次出现顺序）
    QStringList groups;
    for (const ParamSchema &s : m_schemas) {
        const QString g = s.group.trimmed().isEmpty() ? tr("常规") : s.group.trimmed();
        if (!groups.contains(g))
            groups.append(g);
    }

    auto *outer = new QVBoxLayout(this);
    outer->setSpacing(8);

    for (const QString &group : std::as_const(groups)) {
        auto *box = new QGroupBox(group, this);
        auto *form = new QFormLayout(box);
        form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

        for (const ParamSchema &s : m_schemas) {
            const QString g = s.group.trimmed().isEmpty() ? tr("常规") : s.group.trimmed();
            if (g != group)
                continue;

            QWidget *editor = createEditor(s);
            form->addRow(s.name, editor);
            if (!s.tooltip.isEmpty()) {
                auto *tip = new QLabel(QStringLiteral("<span style=\"color:#888\">%1</span>")
                                           .arg(s.tooltip.toHtmlEscaped()), box);
                tip->setWordWrap(true);
                form->addRow(QString(), tip);
            }
        }
        outer->addWidget(box);
    }
    outer->addStretch();
}

QJsonObject ParamFormGenerator::collectParams() const
{
    QJsonObject out;
    for (const ParamSchema &s : m_schemas) {
        QWidget *editor = findChild<QWidget *>(s.name);
        if (!editor) {
            out.insert(s.name, QJsonValue::fromVariant(s.defaultValue));
            continue;
        }

        switch (s.type) {
        case ParamSchema::Type::Bool: {
            auto *cb = qobject_cast<QCheckBox *>(editor);
            out.insert(s.name, cb ? cb->isChecked() : s.defaultValue.toBool());
            break;
        }
        case ParamSchema::Type::Enum: {
            auto *combo = qobject_cast<QComboBox *>(editor);
            if (combo && combo->count() > 0 && combo->currentIndex() >= 0)
                out.insert(s.name, combo->currentData().toString());
            else
                out.insert(s.name, QJsonValue::fromVariant(s.defaultValue));
            break;
        }
        case ParamSchema::Type::Int: {
            auto *spin = qobject_cast<QSpinBox *>(editor);
            out.insert(s.name, spin ? spin->value() : s.defaultValue.toInt());
            break;
        }
        case ParamSchema::Type::Double: {
            auto *dspin = qobject_cast<QDoubleSpinBox *>(editor);
            out.insert(s.name, dspin ? dspin->value() : s.defaultValue.toDouble());
            break;
        }
        case ParamSchema::Type::Path:
        case ParamSchema::Type::String:
        default: {
            // String 的编辑器直接是 QLineEdit；Path 是容器（内嵌 QLineEdit）
            auto *edit = qobject_cast<QLineEdit *>(editor);
            if (!edit)
                edit = editor->findChild<QLineEdit *>();
            if (edit)
                out.insert(s.name, edit->text());
            else
                out.insert(s.name, s.defaultValue.toString());
            break;
        }
        }
    }
    return out;
}

void ParamFormGenerator::setParams(const QJsonObject &params)
{
    for (const ParamSchema &s : m_schemas) {
        QWidget *editor = findChild<QWidget *>(s.name);
        if (!editor)
            continue;
        const QVariant v = params.contains(s.name) ? params.value(s.name).toVariant()
                                                   : s.defaultValue;

        switch (s.type) {
        case ParamSchema::Type::Bool: {
            if (auto *cb = qobject_cast<QCheckBox *>(editor)) {
                cb->blockSignals(true);
                cb->setChecked(v.toBool());
                cb->blockSignals(false);
            }
            break;
        }
        case ParamSchema::Type::Enum: {
            if (auto *combo = qobject_cast<QComboBox *>(editor)) {
                combo->blockSignals(true);
                int idx = combo->findData(v.toString());
                if (idx >= 0)
                    combo->setCurrentIndex(idx);
                combo->blockSignals(false);
            }
            break;
        }
        case ParamSchema::Type::Int: {
            if (auto *spin = qobject_cast<QSpinBox *>(editor)) {
                spin->blockSignals(true);
                spin->setValue(v.toInt());
                spin->blockSignals(false);
            }
            break;
        }
        case ParamSchema::Type::Double: {
            if (auto *dspin = qobject_cast<QDoubleSpinBox *>(editor)) {
                dspin->blockSignals(true);
                dspin->setValue(v.toDouble());
                dspin->blockSignals(false);
            }
            break;
        }
        case ParamSchema::Type::Path:
        case ParamSchema::Type::String:
        default: {
            auto *edit = qobject_cast<QLineEdit *>(editor);
            if (!edit)
                edit = editor->findChild<QLineEdit *>();
            if (edit) {
                edit->blockSignals(true);
                edit->setText(v.toString());
                edit->blockSignals(false);
            }
            break;
        }
        }
    }
}

QWidget *ParamFormGenerator::createEditor(const ParamSchema &s)
{
    switch (s.type) {
    case ParamSchema::Type::Int: {
        auto *spin = new QSpinBox(this);
        spin->setObjectName(s.name);
        spin->setRange((s.min.isValid() && s.min.canConvert<int>()) ? s.min.toInt()
                                                                   : std::numeric_limits<int>::min(),
                       (s.max.isValid() && s.max.canConvert<int>()) ? s.max.toInt()
                                                                   : std::numeric_limits<int>::max());
        spin->setValue(s.defaultValue.toInt());
        // 悬停滚轮不改值：必须去掉 WheelFocus（默认策略下滚轮会先给控件
        // 焦点，焦点门控过滤器就被绕过了），配合 ParamWheelFilter 使用
        spin->setFocusPolicy(Qt::StrongFocus);
        spin->installEventFilter(new ParamWheelFilter(spin));
        QObject::connect(spin, QOverload<int>::of(&QSpinBox::valueChanged),
                         this, &ParamFormGenerator::valuesChanged);
        return spin;
    }
    case ParamSchema::Type::Double: {
        auto *spin = new QDoubleSpinBox(this);
        spin->setObjectName(s.name);
        spin->setDecimals(3);
        spin->setRange((s.min.isValid() && s.min.canConvert<double>()) ? s.min.toDouble() : -1e6,
                       (s.max.isValid() && s.max.canConvert<double>()) ? s.max.toDouble() : 1e6);
        spin->setValue(s.defaultValue.toDouble());
        // 悬停滚轮不改值：必须去掉 WheelFocus（默认策略下滚轮会先给控件
        // 焦点，焦点门控过滤器就被绕过了），配合 ParamWheelFilter 使用
        spin->setFocusPolicy(Qt::StrongFocus);
        spin->installEventFilter(new ParamWheelFilter(spin));
        QObject::connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                         this, &ParamFormGenerator::valuesChanged);
        return spin;
    }
    case ParamSchema::Type::Bool: {
        auto *cb = new QCheckBox(this);
        cb->setObjectName(s.name);
        cb->setChecked(s.defaultValue.toBool());
        QObject::connect(cb, &QCheckBox::toggled, this, &ParamFormGenerator::valuesChanged);
        return cb;
    }
    case ParamSchema::Type::Enum: {
        auto *combo = new QComboBox(this);
        combo->setObjectName(s.name);
        for (const QString &v : s.enumValues)
            combo->addItem(v, v);
        const int idx = combo->findData(s.defaultValue.toString());
        if (idx >= 0)
            combo->setCurrentIndex(idx);
        // 悬停滚轮不切换：必须去掉 WheelFocus（默认策略下滚轮会先给控件
        // 焦点，焦点门控过滤器就被绕过了），配合 ParamWheelFilter 使用
        combo->setFocusPolicy(Qt::StrongFocus);
        combo->installEventFilter(new ParamWheelFilter(combo));
        QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                         this, &ParamFormGenerator::valuesChanged);
        return combo;
    }
    case ParamSchema::Type::Path: {
        auto *container = new QWidget(this);
        auto *h = new QHBoxLayout(container);
        h->setContentsMargins(0, 0, 0, 0);
        auto *edit = new QLineEdit(container);
        edit->setObjectName(s.name);
        if (s.defaultValue.canConvert<QString>())
            edit->setText(s.defaultValue.toString());
        auto *btn = new QToolButton(container);
        btn->setText(QStringLiteral("..."));
        btn->setToolTip(tr("浏览..."));
        QObject::connect(btn, &QToolButton::clicked, this, [this, edit]() {
            const QString fn = QFileDialog::getOpenFileName(this, tr("选择文件"), edit->text());
            if (!fn.isEmpty()) {
                edit->setText(fn);
                emit valuesChanged();
            }
        });
        h->addWidget(edit, 1);
        h->addWidget(btn);
        QObject::connect(edit, &QLineEdit::textChanged,
                         this, &ParamFormGenerator::valuesChanged);
        return container;
    }
    case ParamSchema::Type::String:
    default: {
        auto *edit = new QLineEdit(this);
        edit->setObjectName(s.name);
        if (s.defaultValue.canConvert<QString>())
            edit->setText(s.defaultValue.toString());
        QObject::connect(edit, &QLineEdit::textChanged,
                         this, &ParamFormGenerator::valuesChanged);
        return edit;
    }
    }
}