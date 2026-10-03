#include "DenseReconstructionFactory.h"

#include "PatchMatchReconstructor.h"
#include "PatchMatchReconstructorV2.h"
#include "StereoSGBMReconstructor.h"

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

void DenseReconstructionFactory::registerBuiltinMethods()
{
    // 幂等注册（静态库链接时，实现 TU 中的静态注册对象可能不被拉入，
    // 因此内置方法统一在此显式登记，由重建管线在运行前调用）。
    if (m_creators.contains("StereoSGBM")) {
        return;
    }
    registerMethod("StereoSGBM", "SGBM 立体匹配",
                   []() -> IDenseReconstructorPtr {
                       return std::make_unique<StereoSGBMReconstructor>();
                   });
    registerMethod("PatchMatch", "PatchMatch 多视图匹配",
                   []() -> IDenseReconstructorPtr {
                       return std::make_unique<PatchMatchReconstructor>();
                   });
    registerMethod("PatchMatchV2", "PatchMatch 优化版（多尺度+平面先验）",
                   []() -> IDenseReconstructorPtr {
                       return std::make_unique<PatchMatchReconstructorV2>();
                   });
}