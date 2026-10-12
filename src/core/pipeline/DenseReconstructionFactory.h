#ifndef DENSERECONSTRUCTIONFACTORY_H
#define DENSERECONSTRUCTIONFACTORY_H

#include <QList>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVariant>
#include <functional>
#include <memory>

#include "plugin/ParamSchema.h"

class IDenseReconstructor;
using IDenseReconstructorPtr = std::unique_ptr<IDenseReconstructor>;

/**
 * @brief 密集重建方法工厂（单例）
 *
 * 头文件刻意不包含 IDenseReconstructor 完整定义，避免 UI 层经本工厂
 * 间接引入 PCL。算法侧通过 create() 拿到完整接口；UI 侧用
 * paramSchemaFor / validateParamsFor 消费参数契约。
 */
class DenseReconstructionFactory
{
public:
    using Creator = std::function<IDenseReconstructorPtr()>;

    static DenseReconstructionFactory &instance();

    /// 注册一种密集重建方法（name 为唯一标识，displayName 用于 UI 展示）
    void registerMethod(const QString &name, const QString &displayName, Creator creator);

    /// 创建指定名称的方法实例；若未注册返回 nullptr
    /// （调用方所在 TU 需包含 IDenseReconstructor.h，以便销毁 unique_ptr）
    IDenseReconstructorPtr create(const QString &name) const;

    /// 获取所有已注册方法的名称列表
    QStringList availableMethods() const;

    /// 获取所有已注册方法的显示名称列表
    QStringList availableDisplayNames() const;

    /// 查询方法参数协议（无实例则返回空列表）
    QList<ParamSchema> paramSchemaFor(const QString &name) const;

    /// 按方法 schema 校验参数
    bool validateParamsFor(const QString &name,
                           const QMap<QString, QVariant> &params,
                           QString *errorMsg = nullptr) const;

    /// 显式注册内置方法（PatchMatch / PatchMatchV2），幂等。
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
 * @brief 自动注册辅助模板（仅算法侧 .cpp 使用）
 *
 * 在实现类的 .cpp 文件中定义一个静态对象即可自动注册。
 * 示例：
 *   static DenseReconstructionRegistrar<PatchMatchReconstructor> registrar;
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
