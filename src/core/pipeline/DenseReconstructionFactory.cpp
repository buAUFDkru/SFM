#include "DenseReconstructionFactory.h"
#include "IDenseReconstructor.h"
#include "PatchMatchReconstructor.h"
#include "PatchMatchReconstructorV2.h"

DenseReconstructionFactory &DenseReconstructionFactory::instance()
{
    static DenseReconstructionFactory inst;
    return inst;
}

void DenseReconstructionFactory::registerMethod(
    const QString &name,
    const QString &displayName,
    Creator creator)
{
    if (name.isEmpty() || !creator) {
        return;
    }
    m_creators[name] = std::move(creator);
    m_displayNames[name] = displayName.isEmpty() ? name : displayName;
}

IDenseReconstructorPtr DenseReconstructionFactory::create(const QString &name) const
{
    auto it = m_creators.find(name);
    if (it == m_creators.end()) {
        return nullptr;
    }
    return it.value()();
}

QStringList DenseReconstructionFactory::availableMethods() const
{
    return m_creators.keys();
}

QStringList DenseReconstructionFactory::availableDisplayNames() const
{
    QStringList names;
    names.reserve(m_creators.size());
    for (const auto &name : m_creators.keys()) {
        auto it = m_displayNames.find(name);
        names.append(it != m_displayNames.end() ? it.value() : name);
    }
    return names;
}

QList<ParamSchema> DenseReconstructionFactory::paramSchemaFor(const QString &name) const
{
    auto inst = create(name);
    return inst ? inst->paramSchema() : QList<ParamSchema>{};
}

bool DenseReconstructionFactory::validateParamsFor(
    const QString &name,
    const QMap<QString, QVariant> &params,
    QString *errorMsg) const
{
    auto inst = create(name);
    if (!inst) {
        if (errorMsg)
            *errorMsg = QStringLiteral("未知密集重建方法: %1").arg(name);
        return false;
    }
    return inst->validateParams(params, errorMsg);
}

void DenseReconstructionFactory::registerBuiltinMethods()
{
    // 幂等注册（静态库链接时，实现 TU 中的静态注册对象可能不被拉入，
    // 因此内置方法统一在此显式登记，由重建管线在运行前调用）。
    if (m_creators.contains("PatchMatch")) {
        return;
    }
    // ADR-004 / F15a：V1 为 Deprecated 对照预设；V2 为推荐预设（策略内核抽取进行中）
    registerMethod("PatchMatch", "PatchMatch 多视图匹配 (Deprecated)",
                   []() -> IDenseReconstructorPtr {
                       return std::make_unique<PatchMatchReconstructor>();
                   });
    registerMethod("PatchMatchV2", "PatchMatch 优化版（多尺度+平面先验）",
                   []() -> IDenseReconstructorPtr {
                       return std::make_unique<PatchMatchReconstructorV2>();
                   });
}