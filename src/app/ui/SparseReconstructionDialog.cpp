/**
 * @file SparseReconstructionDialog.cpp
 * @brief 稀疏重建配置对话框实现（非模态）
 */

#include "SparseReconstructionDialog.h"
#include "ControlButtons.h"
#include "ParamWidget/AlgorithmComboPanel.h"

#include <QCloseEvent>
#include <QFrame>
#include <QScrollArea>
#include <QScreen>
#include <QVBoxLayout>

SparseReconstructionDialog::SparseReconstructionDialog(PluginRegistry *registry,
                                                       QWidget *parent)
    : QDialog(parent)
    , m_registry(registry)
{
    setupUI();

    // 默认算法选择：描述符 SIFT、匹配器 FLANN（与历史默认一致）。
    // 参数传空对象 → 表单按各插件 paramSchema 的默认值填充。
    m_descriptor->setParams(QStringLiteral("SIFT"), QJsonObject{});
    m_matcher->setParams(QStringLiteral("FLANN"), QJsonObject{});

    // 未加载项目时禁用开始按钮
    setStartEnabled(false);

    connect(m_controls, &ControlButtons::startClicked,
            this, &SparseReconstructionDialog::startRequested);
    connect(m_controls, &ControlButtons::stopClicked,
            this, &SparseReconstructionDialog::stopRequested);
    connect(m_descriptor, &AlgorithmComboPanel::changed,
            this, &SparseReconstructionDialog::configChanged);
    connect(m_matcher, &AlgorithmComboPanel::changed,
            this, &SparseReconstructionDialog::configChanged);
}

void SparseReconstructionDialog::setupUI()
{
    setWindowTitle(tr("稀疏重建"));
    setModal(false);
    setMinimumWidth(440);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    // 参数表单放入滚动区域：算法参数较多、表单较高，
    // 超出可视高度时用滚轮滚动（与密集重建对话框同一模式）
    auto *scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    auto *content = new QWidget(scroll);
    auto *formLayout = new QVBoxLayout(content);
    formLayout->setSpacing(8);
    formLayout->setContentsMargins(12, 12, 12, 12);

    m_descriptor = new AlgorithmComboPanel(QStringLiteral("descriptor"),
                                           m_registry, tr("特征描述符"), content);
    formLayout->addWidget(m_descriptor);

    QFrame *line1 = new QFrame(content);
    line1->setFrameShape(QFrame::HLine);
    line1->setFrameShadow(QFrame::Sunken);
    formLayout->addWidget(line1);

    m_matcher = new AlgorithmComboPanel(QStringLiteral("matcher"),
                                        m_registry, tr("特征匹配器"), content);
    formLayout->addWidget(m_matcher);

    QFrame *line2 = new QFrame(content);
    line2->setFrameShape(QFrame::HLine);
    line2->setFrameShadow(QFrame::Sunken);
    formLayout->addWidget(line2);

    m_controls = new ControlButtons(content);
    formLayout->addWidget(m_controls);

    formLayout->addStretch();
    scroll->setWidget(content);
    layout->addWidget(scroll, /*stretch=*/1);

    // 对话框高度不超过屏幕可用高度的 85%，避免超出屏幕；
    // 初始尺寸取最小宽度 440（表单标签+控件需要余量，320 过窄）
    if (const QScreen *scr = this->screen()) {
        const int maxH = static_cast<int>(scr->availableGeometry().height() * 0.85);
        setMaximumHeight(maxH);
        resize(440, qMin(maxH, 640));
    }
}

void SparseReconstructionDialog::gatherAlgorithms(
    QString &descriptorType, QJsonObject &descriptorParams,
    QString &matcherType, QJsonObject &matcherParams) const
{
    const QJsonObject desc = m_descriptor->collectParams();
    descriptorType = desc.value(QStringLiteral("algorithm")).toString();
    descriptorParams = desc.value(QStringLiteral("params")).toObject();

    const QJsonObject matcher = m_matcher->collectParams();
    matcherType = matcher.value(QStringLiteral("algorithm")).toString();
    matcherParams = matcher.value(QStringLiteral("params")).toObject();
}

void SparseReconstructionDialog::setAlgorithms(const QString &descriptorType,
                                               const QJsonObject &descriptorParams,
                                               const QString &matcherType,
                                               const QJsonObject &matcherParams)
{
    m_updating = true;
    m_descriptor->setParams(descriptorType, descriptorParams);
    m_matcher->setParams(matcherType, matcherParams);
    m_updating = false;
}

void SparseReconstructionDialog::setRunning(bool running)
{
    // 运行中锁定算法配置；按钮内部已有运行态门控
    m_descriptor->setEnabled(!running);
    m_matcher->setEnabled(!running);
    m_controls->setRunning(running);
}

void SparseReconstructionDialog::setStartEnabled(bool enabled)
{
    m_controls->setStartEnabled(enabled);
}

void SparseReconstructionDialog::closeEvent(QCloseEvent *event)
{
    // 非模态复用：关闭仅隐藏，实例与运行状态保持
    hide();
    event->ignore();
}
