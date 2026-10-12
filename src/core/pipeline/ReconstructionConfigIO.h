#ifndef RECONSTRUCTIONCONFIGIO_H
#define RECONSTRUCTIONCONFIGIO_H

/**
 * @file ReconstructionConfigIO.h
 * @brief ReconstructionConfig 预设读写（ADR-002）
 *
 * 业务调优默认值外置为 JSON（:/presets/*.json 或磁盘路径），
 * 结构体本身只保留类型与语言级缺省。
 */

#include "pipeline/ReconstructionPipeline.h"

#include <QString>

namespace sfm {

/// 将 JSON 对象中的已知键合并进 config（不覆盖 imageFiles / project* / cameraMatrix）
bool applyReconstructionPreset(ReconstructionConfig &config, const QJsonObject &preset,
                               QString *error = nullptr);

/// 从文件或 Qt 资源加载预设。resourceOrPath 可为 ":/presets/default.json" 或磁盘路径
bool loadReconstructionPreset(ReconstructionConfig &config, const QString &resourceOrPath,
                              QString *error = nullptr);

/// 按预设 id 加载内置资源（default / wide-baseline）
bool loadBuiltinReconstructionPreset(ReconstructionConfig &config, const QString &presetId,
                                     QString *error = nullptr);

/// 返回已应用 default 预设的配置（供 Pipeline / CLI / 测试构造）
ReconstructionConfig makeDefaultReconstructionConfig();

} // namespace sfm

#endif
