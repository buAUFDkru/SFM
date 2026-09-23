#ifndef DENSERECONSTRUCTIONFACTORY_H
#define DENSERECONSTRUCTIONFACTORY_H

#include <QString>
#include <QStringList>
#include <QMap>
#include <functional>
#include <memory>
#include "IDenseReconstructor.h"

/**
 * @brief 密集重建方法工厂（单例）
 *
 * 注册并创建各种密集重建方法实例。
 * 新算法只需在初始化时调用 registerMethod 即可加入列表。
 */
class DenseReconstructionFactory
{
public:
    using Creator = std::function<IDenseReconstructorPtr()>;

    static DenseReconstructionFactory &instance();

    /// 注册一种密集重建方法（name 为唯一标识，displayName 用于 UI 展示）
    void registerMethod(const QString &name, const QString &displayName, Creator creator);

    /// 创建指定名称的方法实例；若未注册返回 nullptr
    IDenseReconstructorPtr create(const QString &name) const;

    /// 获取所有已注册方法的名称列表
    QStringList availableMethods() const;

    /// 获取所有已注册方法的显示名称列表
    QStringList availableDisplayNames() const;

    /// 显式注册内置方法（StereoSGBM / PatchMatch），幂等。
    /// 静态库链接时实现 TU 的静态注册对象可能不会被拉入，因此由重建管线运行前调用。
    void registerBuiltinMethods();

private:
    DenseReconstructionFactory() = default;
    DenseReconstructionFactory(const DenseReconstructionFactory &) = delete;
    DenseReconstructionFactory &operator=(const DenseReconstructionFactory &) = delete;

    QMap<QString, Creator> m_creators;
    QMap<QString, QString> m_displayNames;
};

/**
 * @brief 自动注册辅助宏/模板
 *
 * 在实现类的 .cpp 文件中定义一个静态对象即可自动注册。
 * 示例：
 *   static DenseReconstructionRegistrar<StereoSGBMReconstructor> registrar;
 */
template <typename T>
class DenseReconstructionRegistrar
{
public:
    DenseReconstructionRegistrar()
    {
        T proto;
        DenseReconstructionFactory::instance().registerMethod(
            proto.methodName(),
            proto.displayName(),
            []() -> IDenseReconstructorPtr { return std::make_unique<T>(); });
    }
};

#endif // DENSERECONSTRUCTIONFACTORY_H
