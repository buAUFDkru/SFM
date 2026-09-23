#include "FlannMatcherPlugin.h"

#include <QDebug>

#include "FlannMatcherWorker.h"

QList<ParamSchema> FlannMatcherPlugin::paramSchema() const
{
    ParamSchema trees;
    trees.name = QStringLiteral("trees");
    trees.type = ParamSchema::Type::Int;
    trees.defaultValue = 4;
    trees.min = 1;
    trees.max = 32;
    trees.group = QStringLiteral("FLANN");
    trees.tooltip = QStringLiteral("KD 树的棵数");

    ParamSchema checks;
    checks.name = QStringLiteral("checks");
    checks.type = ParamSchema::Type::Int;
    checks.defaultValue = 50;
    checks.min = 1;
    checks.max = 200;
    checks.group = QStringLiteral("FLANN");
    checks.tooltip = QStringLiteral("搜索时回溯节点次数(预留，当前实现未使用)");

    return { trees, checks };
}

bool FlannMatcherPlugin::validateParams(const QJsonObject &params,
                                        QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("trees") ||
            key == QStringLiteral("checks")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else {
            if (errorMsg)
                *errorMsg = QStringLiteral("未知参数 %1").arg(key);
            return false;
        }
    }
    return true;
}

std::unique_ptr<IMatcherWorker> FlannMatcherPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[FLANN] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<FlannMatcherWorker>(params);
}