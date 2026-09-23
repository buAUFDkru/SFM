#include "DenseReconstructionFactory.h"

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
