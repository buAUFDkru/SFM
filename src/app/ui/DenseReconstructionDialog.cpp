/**
 * @file DenseReconstructionDialog.cpp
 * @brief 密集重建方法选择对话框实现（schema 驱动）
 */

#include "DenseReconstructionDialog.h"
#include "ParamWidget/ParamFormGenerator.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QVBoxLayout>

#include "pipeline/DenseReconstructionFactory.h"

namespace {

/// 为各密集重建方法定义参数 schema（密集方法尚未暴露 paramSchema()，
/// 此处以固定 schema 驱动通用参数表单生成器，新增参数只需维护此表）
QList<ParamSchema> schemasForMethod(const QString &method)
{
    QList<ParamSchema> out;
    if (method == QStringLiteral("StereoSGBM")) {
        ParamSchema maxDepth;
        maxDepth.name = QStringLiteral("maxDepth");
        maxDepth.type = ParamSchema::Type::Double;
        maxDepth.defaultValue = 50.0;
        maxDepth.min = 0.1;
        maxDepth.max = 1000.0;
        maxDepth.group = QStringLiteral("SGBM");
        maxDepth.tooltip = QObject::tr("最大深度（场景单位），超出被置零");
        out.append(maxDepth);

        ParamSchema numDisp;
        numDisp.name = QStringLiteral("numDisparities");
        numDisp.type = ParamSchema::Type::Int;
        numDisp.defaultValue = 64;
        numDisp.min = 16;
        numDisp.max = 1024;
        numDisp.group = QStringLiteral("SGBM");
        numDisp.tooltip = QObject::tr("视差搜索范围（16 的倍数），越大越慢");
        out.append(numDisp);

        ParamSchema blockSize;
        blockSize.name = QStringLiteral("blockSize");
        blockSize.type = ParamSchema::Type::Int;
        blockSize.defaultValue = 5;
        blockSize.min = 1;
        blockSize.max = 21;
        blockSize.group = QStringLiteral("SGBM");
        blockSize.tooltip = QObject::tr("匹配块大小（奇数，3~11 推荐）");
        out.append(blockSize);

        ParamSchema minDisp;
        minDisp.name = QStringLiteral("minDisparity");
        minDisp.type = ParamSchema::Type::Int;
        minDisp.defaultValue = 0;
        minDisp.min = -1024;
        minDisp.max = 1024;
        minDisp.group = QStringLiteral("SGBM");
        out.append(minDisp);

        ParamSchema useHH;
        useHH.name = QStringLiteral("useHHMode");
        useHH.type = ParamSchema::Type::Bool;
        useHH.defaultValue = false;
        useHH.group = QStringLiteral("SGBM");
        useHH.tooltip = QObject::tr("使用 HH 全分辨率模式（更精确但更慢）");
        out.append(useHH);

        ParamSchema useWls;
        useWls.name = QStringLiteral("useWLSFilter");
        useWls.type = ParamSchema::Type::Bool;
        useWls.defaultValue = false;
        useWls.group = QStringLiteral("SGBM");
        useWls.tooltip = QObject::tr("启用 WLS 权重最小二乘后滤波（edge 保持）");
        out.append(useWls);
    }
    // PatchMatch 等其余方法暂无可配置参数
    return out;
}

} // namespace

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
    setMinimumWidth(380);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    QFormLayout *formLayout = new QFormLayout();
    m_methodCombo = new QComboBox(this);
    formLayout->addRow(tr("密集重建方法:"), m_methodCombo);
    mainLayout->addLayout(formLayout);

    m_form = new ParamFormGenerator(this);
    mainLayout->addWidget(m_form);

    m_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    mainLayout->addWidget(m_buttonBox);

    connect(m_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseReconstructionDialog::updateParamsForCurrentMethod);
    connect(m_buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
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

    // 方法切换前缓存当前方法参数
    const QString method = m_methodCombo->currentData().toString();
    m_methodParams[method] = selectedParams();

    // 重建当前方法表单
    m_form->setSchemas(schemasForMethod(method));
    m_form->setParams(QJsonObject::fromVariantMap(m_methodParams.value(method)));
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