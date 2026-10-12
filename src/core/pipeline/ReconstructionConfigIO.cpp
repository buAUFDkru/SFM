#include "pipeline/ReconstructionConfigIO.h"

#include <QDebug>
#include <QFile>
#include <QIODevice>
#include <QJsonDocument>
#include <QJsonParseError>

namespace sfm {
namespace {

/// 静态库中的 qrc 不会被可执行文件自动注册，需显式初始化
void ensurePresetsResource()
{
    static const bool once = [] {
        Q_INIT_RESOURCE(presets);
        return true;
    }();
    Q_UNUSED(once);
}

bool readJsonObject(const QString &resourceOrPath, QJsonObject *out, QString *error)
{
    if (resourceOrPath.startsWith(QLatin1Char(':')))
        ensurePresetsResource();

    QFile f(resourceOrPath);
    if (!f.open(QIODevice::ReadOnly)) {
        if (error)
            *error = QStringLiteral("无法打开预设: %1").arg(resourceOrPath);
        return false;
    }
    QJsonParseError pe;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &pe);
    if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
        if (error)
            *error = QStringLiteral("预设 JSON 无效: %1 (%2)")
                         .arg(resourceOrPath, pe.errorString());
        return false;
    }
    *out = doc.object();
    return true;
}

} // namespace

bool applyReconstructionPreset(ReconstructionConfig &config, const QJsonObject &preset,
                               QString *error)
{
    if (preset.isEmpty()) {
        if (error)
            *error = QStringLiteral("预设对象为空");
        return false;
    }

    if (preset.contains(QStringLiteral("descriptorType")))
        config.descriptorType = preset.value(QStringLiteral("descriptorType")).toString();
    if (preset.contains(QStringLiteral("descriptorParams")) &&
        preset.value(QStringLiteral("descriptorParams")).isObject())
        config.descriptorParams = preset.value(QStringLiteral("descriptorParams")).toObject();

    if (preset.contains(QStringLiteral("matcherType")))
        config.matcherType = preset.value(QStringLiteral("matcherType")).toString();
    if (preset.contains(QStringLiteral("matcherParams")) &&
        preset.value(QStringLiteral("matcherParams")).isObject())
        config.matcherParams = preset.value(QStringLiteral("matcherParams")).toObject();

    if (preset.contains(QStringLiteral("threadCount")))
        config.threadCount = preset.value(QStringLiteral("threadCount")).toInt();
    if (preset.contains(QStringLiteral("baMaxIterations")))
        config.baMaxIterations = preset.value(QStringLiteral("baMaxIterations")).toInt();
    if (preset.contains(QStringLiteral("voxelSize")))
        config.voxelSize = preset.value(QStringLiteral("voxelSize")).toDouble();
    if (preset.contains(QStringLiteral("useGpu")))
        config.useGpu = preset.value(QStringLiteral("useGpu")).toBool();
    if (preset.contains(QStringLiteral("useExifFocal")))
        config.useExifFocal = preset.value(QStringLiteral("useExifFocal")).toBool();
    if (preset.contains(QStringLiteral("refineDistortion")))
        config.refineDistortion = preset.value(QStringLiteral("refineDistortion")).toBool();
    if (preset.contains(QStringLiteral("refineFocalPerAxis")))
        config.refineFocalPerAxis = preset.value(QStringLiteral("refineFocalPerAxis")).toBool();

    if (preset.contains(QStringLiteral("denseReconstructionMethod")))
        config.denseReconstructionMethod =
            preset.value(QStringLiteral("denseReconstructionMethod")).toString();

    return true;
}

bool loadReconstructionPreset(ReconstructionConfig &config, const QString &resourceOrPath,
                              QString *error)
{
    QJsonObject obj;
    if (!readJsonObject(resourceOrPath, &obj, error))
        return false;
    return applyReconstructionPreset(config, obj, error);
}

bool loadBuiltinReconstructionPreset(ReconstructionConfig &config, const QString &presetId,
                                     QString *error)
{
    const QString path = QStringLiteral(":/presets/%1.json").arg(presetId);
    return loadReconstructionPreset(config, path, error);
}

ReconstructionConfig makeDefaultReconstructionConfig()
{
    ReconstructionConfig cfg;
    QString err;
    if (!loadBuiltinReconstructionPreset(cfg, QStringLiteral("default"), &err)) {
        // 资源缺失时保留语言级缺省，避免构造失败
        qWarning("[ReconstructionConfig] 加载 default 预设失败: %s", qPrintable(err));
    }
    return cfg;
}

} // namespace sfm
