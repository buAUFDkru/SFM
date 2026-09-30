/**
 * @file AlgorithmComboPanel.cpp
 * @brief 算法选择面板实现（schema 驱动）
 */

#include "AlgorithmComboPanel.h"
#include "ParamFormGenerator.h"

#include <QComboBox>
#include <QEvent>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

#include "plugin/ISfmPlugin.h"
#include "plugin/PluginRegistry.h"

namespace {

/// 拦截 QComboBox 本体的滚轮事件：参数表单内滚动滚轮应滚动页面而非切换算法，
/// 仅点击下拉展开后才能切换（弹出列表自身的滚轮浏览不受影响）
class ComboWheelFilter : public QObject
{
public:
    explicit ComboWheelFilter(QObject *parent) : QObject(parent) {}

    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (event->type() == QEvent::Wheel)
            return true;
        return QObject::eventFilter(watched, event);
    }
};

} // namespace

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
    // 滚轮不切换算法（以 combo 为 parent，随其销毁）
    m_algoCombo->installEventFilter(new ComboWheelFilter(m_algoCombo));
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

    // 初始填充下拉框发生在 connect 之前，currentIndexChanged 不会触发，
    // 必须显式构建一次表单，否则启动界面不显示当前算法的参数
    rebuildForm();
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
    // 索引实际变化时由 currentIndexChanged 触发重建，避免双重 rebuild
    bool rebuilt = false;
    if (m_registry) {
        const int idx = m_algoCombo->findData(algorithm);
        if (idx >= 0 && idx != m_algoCombo->currentIndex()) {
            m_algoCombo->setCurrentIndex(idx);
            rebuilt = true;
        }
    }
    if (!rebuilt)
        rebuildForm();
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