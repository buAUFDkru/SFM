/**
 * @file AlgorithmComboPanel.cpp
 * @brief 算法选择面板实现（schema 驱动）
 */

#include "AlgorithmComboPanel.h"
#include "ParamFormGenerator.h"

#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

#include "plugin/ISfmPlugin.h"
#include "plugin/PluginRegistry.h"

AlgorithmComboPanel::AlgorithmComboPanel(const QString &category,
                                         PluginRegistry *registry,
                                         const QString &title,
                                         QWidget *parent)
    : QWidget(parent)
    , m_category(category)
    , m_registry(registry)
    , m_title(title)
{
    auto *box = new QGroupBox(m_title, this);
    auto *v = new QVBoxLayout(box);

    // 算法下拉框
    auto *row = new QHBoxLayout();
    row->addWidget(new QLabel(tr("算法:"), box));
    m_algoCombo = new QComboBox(box);
    row->addWidget(m_algoCombo, 1);
    v->addLayout(row);

    // 参数表单占位
    m_form = new ParamFormGenerator(box);
    v->addWidget(m_form);

    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->addWidget(box);

    // 从注册表填充算法列表
    if (m_registry) {
        const QList<PluginInfo> infos = m_registry->byCategory(m_category);
        for (const PluginInfo &info : infos)
            m_algoCombo->addItem(info.name, info.id);
    }

    connect(m_algoCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                rebuildForm();
                emit changed();
            });
    connect(m_form, &ParamFormGenerator::valuesChanged, this, &AlgorithmComboPanel::changed);
}

QString AlgorithmComboPanel::currentAlgorithm() const
{
    const int idx = m_algoCombo->currentIndex();
    if (idx < 0)
        return QString();
    return m_algoCombo->itemData(idx).toString();
}

QJsonObject AlgorithmComboPanel::collectParams() const
{
    QJsonObject out{
        {"algorithm", currentAlgorithm()},
        {"params", m_form->collectParams()},
    };
    return out;
}

void AlgorithmComboPanel::setParams(const QString &algorithm, const QJsonObject &params)
{
    if (m_registry) {
        const int idx = m_algoCombo->findData(algorithm);
        if (idx >= 0) {
            m_algoCombo->setCurrentIndex(idx);
            rebuildForm();
        }
    }
    m_form->setParams(params);
    emit changed();
}

void AlgorithmComboPanel::rebuildForm()
{
    if (!m_registry)
        return;
    const QString id = currentAlgorithm();
    if (id.isEmpty()) {
        m_form->setSchemas({});
        return;
    }
    ISfmPlugin *plugin = m_registry->plugin(id);
    if (!plugin) {
        m_form->setSchemas({});
        return;
    }
    m_form->setSchemas(plugin->paramSchema());
}