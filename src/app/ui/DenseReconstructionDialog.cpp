/**
 * @file DenseReconstructionDialog.cpp
 * @brief 密集重建方法选择对话框实现（schema 驱动）
 */

#include "DenseReconstructionDialog.h"
#include "ParamWidget/ParamFormGenerator.h"
#include "ParamWidget/ParamWheelFilter.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QScrollArea>
#include <QScreen>
#include <QVBoxLayout>

#include "pipeline/DenseReconstructionFactory.h"

namespace {

/// 为各密集重建方法定义参数 schema（密集方法尚未暴露 paramSchema()，
/// 此处以固定 schema 驱动通用参数表单生成器，新增参数只需维护此表）
QList<ParamSchema> schemasForMethod(const QString &method)
{
    QList<ParamSchema> out;
    auto addInt = [&out](const QString &name, int def, int mn, int mx,
                         const QString &group, const QString &tip) {
        ParamSchema s;
        s.name = name;
        s.type = ParamSchema::Type::Int;
        s.defaultValue = def;
        s.min = mn;
        s.max = mx;
        s.group = group;
        s.tooltip = tip;
        out.append(s);
    };
    auto addDouble = [&out](const QString &name, double def, double mn, double mx,
                            const QString &group, const QString &tip) {
        ParamSchema s;
        s.name = name;
        s.type = ParamSchema::Type::Double;
        s.defaultValue = def;
        s.min = mn;
        s.max = mx;
        s.group = group;
        s.tooltip = tip;
        out.append(s);
    };
    auto addBool = [&out](const QString &name, bool def,
                          const QString &group, const QString &tip) {
        ParamSchema s;
        s.name = name;
        s.type = ParamSchema::Type::Bool;
        s.defaultValue = def;
        s.group = group;
        s.tooltip = tip;
        out.append(s);
    };
    if (method == QStringLiteral("PatchMatch")) {
        // 与 PatchMatchReconstructor::Options 的常用字段对齐
        addDouble(QStringLiteral("depthMin"), -1.0, -1.0, 1e6,
                  QStringLiteral("深度"), QObject::tr("最小深度；<0 从稀疏点自动估计"));
        addDouble(QStringLiteral("depthMax"), -1.0, -1.0, 1e6,
                  QStringLiteral("深度"), QObject::tr("最大深度；<0 从稀疏点自动估计"));
        addInt(QStringLiteral("maxImageSize"), 1000, 64, 5000,
               QStringLiteral("性能"), QObject::tr("CPU 实现的工作图像最大边长（缩小可显著提速）"));
        addInt(QStringLiteral("windowRadius"), 5, 1, 20,
               QStringLiteral("性能"), QObject::tr("NCC 匹配窗口半径（半宽）"));
        addInt(QStringLiteral("numSamples"), 15, 1, 100,
               QStringLiteral("性能"), QObject::tr("每像素随机采样数（初始化）"));
        addInt(QStringLiteral("numIterations"), 5, 1, 50,
               QStringLiteral("性能"), QObject::tr("邻域传播迭代轮数"));
        addDouble(QStringLiteral("nccSigma"), 0.6, 0.1, 5.0,
                  QStringLiteral("代价"), QObject::tr("NCC 似然函数扩散系数"));
        addDouble(QStringLiteral("sigmaColor"), 51.0, 1.0, 255.0,
                  QStringLiteral("代价"), QObject::tr("颜色权重 sigma（uint8 图像）"));
        addDouble(QStringLiteral("minTriangulationAngle"), 0.5, 0.1, 10.0,
                  QStringLiteral("代价"), QObject::tr("选源图像最小光轴夹角（度）"));
        addDouble(QStringLiteral("filterMinNcc"), 0.1, 0.0, 1.0,
                  QStringLiteral("过滤"), QObject::tr("过滤最小 NCC 阈值（代价上限）"));
        addDouble(QStringLiteral("filterMinTriangulationAngle"), 3.0, 0.0, 30.0,
                  QStringLiteral("过滤"), QObject::tr("过滤最小三角化角度（度）"));
        addBool(QStringLiteral("filter"), true,
                QStringLiteral("过滤"), QObject::tr("启用置信度过滤"));
        addBool(QStringLiteral("geomConsistency"), false,
                QStringLiteral("过滤"), QObject::tr("几何一致性校验（当前保留未实现）"));
    } else if (method == QStringLiteral("PatchMatchV2")) {
        // 与 PatchMatchReconstructorV2::Options（优化版）的常用字段对齐，
        // 默认值即 2026-10-03 全量验证录定配置（多尺度+几何一致性+topk4
        // +融合+全分辨率 1320+平面先验）
        addDouble(QStringLiteral("depthMin"), -1.0, -1.0, 1e6,
                  QStringLiteral("深度"), QObject::tr("最小深度；<0 从稀疏点自动估计"));
        addDouble(QStringLiteral("depthMax"), -1.0, -1.0, 1e6,
                  QStringLiteral("深度"), QObject::tr("最大深度；<0 从稀疏点自动估计"));
        addInt(QStringLiteral("maxImageSize"), 1320, 0, 8000,
               QStringLiteral("性能"), QObject::tr("工作图像最大边长；0 = 不缩放（全分辨率）"));
        addInt(QStringLiteral("windowRadius"), 5, 1, 20,
               QStringLiteral("性能"), QObject::tr("NCC 匹配窗口半径（半宽）"));
        addInt(QStringLiteral("numSamples"), 15, 1, 100,
               QStringLiteral("性能"), QObject::tr("每像素随机采样数（初始化）"));
        addInt(QStringLiteral("numIterations"), 5, 1, 50,
               QStringLiteral("性能"), QObject::tr("邻域传播迭代轮数（多尺度时为最粗层轮数）"));
        addBool(QStringLiteral("multiscale"), true,
                QStringLiteral("性能"), QObject::tr("多尺度金字塔 + JBU 上采样（提速并支撑高分辨率）"));
        addBool(QStringLiteral("planePrior"), true,
                QStringLiteral("性能"), QObject::tr("平面先验（ACMP 受限后验：弱纹理区域更完整、重投影误差更低，耗时约 +12%）"));
        addInt(QStringLiteral("costTopK"), 4, 1, 10,
               QStringLiteral("代价"), QObject::tr("多视图代价聚合：取最优 k 个源视图代价平均（抗遮挡/高光）"));
        addDouble(QStringLiteral("nccSigma"), 0.6, 0.1, 5.0,
                  QStringLiteral("代价"), QObject::tr("NCC 似然函数扩散系数"));
        addDouble(QStringLiteral("sigmaColor"), 51.0, 1.0, 255.0,
                  QStringLiteral("代价"), QObject::tr("颜色权重 sigma（uint8 图像）"));
        addDouble(QStringLiteral("minTriangulationAngle"), 0.5, 0.1, 10.0,
                  QStringLiteral("代价"), QObject::tr("选源图像最小光轴夹角（度）"));
        addDouble(QStringLiteral("filterMinNcc"), 0.1, 0.0, 1.0,
                  QStringLiteral("过滤"), QObject::tr("过滤最小 NCC 阈值（代价上限）"));
        addDouble(QStringLiteral("filterMinTriangulationAngle"), 3.0, 0.0, 30.0,
                  QStringLiteral("过滤"), QObject::tr("过滤最小三角化角度（度）"));
        addBool(QStringLiteral("filter"), true,
                QStringLiteral("过滤"), QObject::tr("启用置信度过滤"));
        addBool(QStringLiteral("geomConsistency"), true,
                QStringLiteral("过滤"), QObject::tr("几何一致性过滤（往返重投影剔除离群点）"));
        addBool(QStringLiteral("fusion"), true,
                QStringLiteral("过滤"), QObject::tr("深度融合：跨视图去重 + 置信度加权平均（降离群点、点云更紧致）"));
    }
    // 其余方法暂无可配置参数
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

    // 限制对话框高度不超过屏幕可用高度的 85%，避免超出屏幕
    if (const QScreen *scr = this->screen()) {
        const int maxH = static_cast<int>(scr->availableGeometry().height() * 0.85);
        setMaximumHeight(maxH);
        resize(minimumSizeHint().expandedTo(QSize(minimumWidth(), maxH / 2)));
    }

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