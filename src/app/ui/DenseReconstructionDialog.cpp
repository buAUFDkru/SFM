/**
 * @file DenseReconstructionDialog.cpp
 * @brief 密集重建方法选择对话框实现（经工厂消费 paramSchema，不触碰 PCL）
 */

#include "DenseReconstructionDialog.h"
#include "ParamWidget/ParamFormGenerator.h"
#include "ParamWidget/ParamWheelFilter.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QJsonObject>
#include <QMessageBox>
#include <QScrollArea>
#include <QScreen>
#include <QVBoxLayout>

#include "pipeline/DenseReconstructionFactory.h"

DenseReconstructionDialog::DenseReconstructionDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUI();
    rebuildMethods();

    if (m_methodCombo->count() > 0) {
        m_methodCombo->setCurrentIndex(0);
        updateParamsForCurrentMethod();
    }
}

void DenseReconstructionDialog::setupUI()
{
    setWindowTitle(tr("选择密集重建方法"));
    setMinimumWidth(440);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    QFormLayout *formLayout = new QFormLayout();
    m_methodCombo = new QComboBox(this);
    // 悬停滚轮不切换方法（焦点门控：需去掉 WheelFocus，否则滚轮先给焦点
    // 绕过过滤器；滚轮留给页面滚动）
    m_methodCombo->setFocusPolicy(Qt::StrongFocus);
    m_methodCombo->installEventFilter(new ParamWheelFilter(m_methodCombo));
    formLayout->addRow(tr("密集重建方法:"), m_methodCombo);
    mainLayout->addLayout(formLayout);

    // 参数表单放入滚动区域：PatchMatch 等方法参数较多，
    // 超出可视高度时用滚轮滚动，方法选择与按钮栏保持可见
    auto *scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_form = new ParamFormGenerator(this);
    scroll->setWidget(m_form);
    mainLayout->addWidget(scroll, /*stretch=*/1);

    m_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    mainLayout->addWidget(m_buttonBox);

    // 限制对话框高度不超过屏幕可用高度的 85%，避免超出屏幕；
    // 初始尺寸与稀疏重建对话框一致（最小宽度 440，初始高 640）
    if (const QScreen *scr = this->screen()) {
        const int maxH = static_cast<int>(scr->availableGeometry().height() * 0.85);
        setMaximumHeight(maxH);
        resize(440, qMin(maxH, 640));
    }

    connect(m_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseReconstructionDialog::updateParamsForCurrentMethod);
    connect(m_buttonBox, &QDialogButtonBox::accepted,
            this, &DenseReconstructionDialog::onAccepted);
    connect(m_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    if (DenseReconstructionFactory::instance().availableMethods().isEmpty())
        DenseReconstructionFactory::instance().registerBuiltinMethods();
}

void DenseReconstructionDialog::rebuildMethods()
{
    m_methodCombo->clear();

    const QStringList methods = DenseReconstructionFactory::instance().availableMethods();
    for (const QString &name : methods) {
        const QString display = DenseReconstructionFactory::instance()
                                    .availableDisplayNames().value(methods.indexOf(name), name);
        m_methodCombo->addItem(display, name);
    }
}

void DenseReconstructionDialog::updateParamsForCurrentMethod()
{
    const int idx = m_methodCombo->currentIndex();
    if (idx < 0)
        return;

    const QString method = m_methodCombo->currentData().toString();
    // combo 已切到新方法；缓存应写回旧方法，避免污染新方法的参数表
    if (!m_currentMethod.isEmpty() && m_currentMethod != method)
        m_methodParams[m_currentMethod] = selectedParams();

    const QList<ParamSchema> schemas =
        DenseReconstructionFactory::instance().paramSchemaFor(method);
    m_form->setSchemas(schemas);
    m_form->setParams(QJsonObject::fromVariantMap(m_methodParams.value(method)));
    m_currentMethod = method;
}

void DenseReconstructionDialog::onAccepted()
{
    const QString method = selectedMethod();
    QString err;
    if (!DenseReconstructionFactory::instance().validateParamsFor(
            method, selectedParams(), &err)) {
        QMessageBox::warning(this, tr("参数无效"), err);
        return;
    }
    accept();
}

QString DenseReconstructionDialog::selectedMethod() const
{
    const int idx = m_methodCombo->currentIndex();
    if (idx < 0)
        return QString();
    return m_methodCombo->itemData(idx).toString();
}

QMap<QString, QVariant> DenseReconstructionDialog::selectedParams() const
{
    return m_form->collectParams().toVariantMap();
}
