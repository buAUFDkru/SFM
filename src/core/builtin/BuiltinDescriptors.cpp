#include "BuiltinDescriptors.h"

#include <QDebug>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

// ── SIFT ────────────────────────────────────────────────────────────────────

SiftDescriptorWorker::SiftDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("nOctaveLayers")))
        m_nOctaveLayers = params.value(QStringLiteral("nOctaveLayers")).toInt(3);
    if (params.contains(QStringLiteral("contrastThreshold")))
        m_contrastThreshold = params.value(QStringLiteral("contrastThreshold")).toDouble(0.04);
    if (params.contains(QStringLiteral("edgeThreshold")))
        m_edgeThreshold = params.value(QStringLiteral("edgeThreshold")).toDouble(10.0);
    if (params.contains(QStringLiteral("sigma")))
        m_sigma = params.value(QStringLiteral("sigma")).toDouble(1.6);
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(8192);
}

FeatureResult SiftDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    auto sift = cv::SIFT::create(m_maxFeatures, m_nOctaveLayers,
                                 m_contrastThreshold, m_edgeThreshold, m_sigma);

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    sift->detectAndCompute(image.imageGray, cv::noArray(),
                           r.keypoints, r.descriptors);
    r.elapsedMs = 0.0;
    return r;
}

QList<ParamSchema> SiftDescriptorPlugin::paramSchema() const
{
    ParamSchema nOctaveLayers;
    nOctaveLayers.name = QStringLiteral("nOctaveLayers");
    nOctaveLayers.type = ParamSchema::Type::Int;
    nOctaveLayers.defaultValue = 3;
    nOctaveLayers.min = 1;
    nOctaveLayers.max = 10;
    nOctaveLayers.group = QStringLiteral("SIFT");

    ParamSchema contrastThreshold;
    contrastThreshold.name = QStringLiteral("contrastThreshold");
    contrastThreshold.type = ParamSchema::Type::Double;
    contrastThreshold.defaultValue = 0.04;
    contrastThreshold.min = 0.0;
    contrastThreshold.max = 1.0;
    contrastThreshold.group = QStringLiteral("SIFT");
    contrastThreshold.tooltip = QStringLiteral("对比度阈值，过滤低对比度关键点");

    ParamSchema edgeThreshold;
    edgeThreshold.name = QStringLiteral("edgeThreshold");
    edgeThreshold.type = ParamSchema::Type::Double;
    edgeThreshold.defaultValue = 10.0;
    edgeThreshold.min = 1.0;
    edgeThreshold.max = 100.0;
    edgeThreshold.group = QStringLiteral("SIFT");
    edgeThreshold.tooltip = QStringLiteral("边缘阈值，过滤位于边缘的关键点");

    ParamSchema sigma;
    sigma.name = QStringLiteral("sigma");
    sigma.type = ParamSchema::Type::Double;
    sigma.defaultValue = 1.6;
    sigma.min = 0.1;
    sigma.max = 10.0;
    sigma.group = QStringLiteral("SIFT");
    sigma.tooltip = QStringLiteral("高斯金字塔初始尺度");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 8192;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("SIFT");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    return { nOctaveLayers, contrastThreshold, edgeThreshold, sigma, maxFeatures };
}

bool SiftDescriptorPlugin::validateParams(const QJsonObject &params,
                                          QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("nOctaveLayers") ||
            key == QStringLiteral("maxFeatures")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("contrastThreshold") ||
                   key == QStringLiteral("edgeThreshold") ||
                   key == QStringLiteral("sigma")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 double").arg(key);
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

std::unique_ptr<IDescriptorWorker> SiftDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[SIFT] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<SiftDescriptorWorker>(params);
}

// ── SURF ────────────────────────────────────────────────────────────────────

SurfDescriptorWorker::SurfDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("hessianThreshold")))
        m_hessianThreshold = params.value(QStringLiteral("hessianThreshold")).toDouble(100.0);
    if (params.contains(QStringLiteral("nOctaves")))
        m_nOctaves = params.value(QStringLiteral("nOctaves")).toInt(4);
    if (params.contains(QStringLiteral("nOctaveLayers")))
        m_nOctaveLayers = params.value(QStringLiteral("nOctaveLayers")).toInt(3);
    if (params.contains(QStringLiteral("extended")))
        m_extended = params.value(QStringLiteral("extended")).toBool(false);
    if (params.contains(QStringLiteral("upright")))
        m_upright = params.value(QStringLiteral("upright")).toBool(false);
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(0);
}

FeatureResult SurfDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    auto surf = cv::xfeatures2d::SURF::create(m_hessianThreshold, m_nOctaves,
                                              m_nOctaveLayers, m_extended, m_upright);

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    surf->detectAndCompute(image.imageGray, cv::noArray(),
                           r.keypoints, r.descriptors);

    // 限制特征点最大数量
    if (m_maxFeatures > 0 &&
        static_cast<int>(r.keypoints.size()) > m_maxFeatures) {
        r.keypoints.resize(m_maxFeatures);
        r.descriptors = r.descriptors.rowRange(0, m_maxFeatures).clone();
    }

    r.elapsedMs = 0.0;
    return r;
}

QList<ParamSchema> SurfDescriptorPlugin::paramSchema() const
{
    ParamSchema hessianThreshold;
    hessianThreshold.name = QStringLiteral("hessianThreshold");
    hessianThreshold.type = ParamSchema::Type::Double;
    hessianThreshold.defaultValue = 100.0;
    hessianThreshold.min = 1.0;
    hessianThreshold.max = 100000.0;
    hessianThreshold.group = QStringLiteral("SURF");
    hessianThreshold.tooltip = QStringLiteral("Hessian 响应阈值，越大提取的点越少");

    ParamSchema nOctaves;
    nOctaves.name = QStringLiteral("nOctaves");
    nOctaves.type = ParamSchema::Type::Int;
    nOctaves.defaultValue = 4;
    nOctaves.min = 1;
    nOctaves.max = 16;
    nOctaves.group = QStringLiteral("SURF");

    ParamSchema nOctaveLayers;
    nOctaveLayers.name = QStringLiteral("nOctaveLayers");
    nOctaveLayers.type = ParamSchema::Type::Int;
    nOctaveLayers.defaultValue = 3;
    nOctaveLayers.min = 1;
    nOctaveLayers.max = 16;
    nOctaveLayers.group = QStringLiteral("SURF");

    ParamSchema extended;
    extended.name = QStringLiteral("extended");
    extended.type = ParamSchema::Type::Bool;
    extended.defaultValue = false;
    extended.group = QStringLiteral("SURF");
    extended.tooltip = QStringLiteral("是否使用扩展(128维)描述子");

    ParamSchema upright;
    upright.name = QStringLiteral("upright");
    upright.type = ParamSchema::Type::Bool;
    upright.defaultValue = false;
    upright.group = QStringLiteral("SURF");
    upright.tooltip = QStringLiteral("是否使用无旋转 Upright 描述子");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 0;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("SURF");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    return { hessianThreshold, nOctaves, nOctaveLayers, extended, upright, maxFeatures };
}

bool SurfDescriptorPlugin::validateParams(const QJsonObject &params,
                                          QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("nOctaves") ||
            key == QStringLiteral("nOctaveLayers") ||
            key == QStringLiteral("maxFeatures")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("hessianThreshold")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 double").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("extended") ||
                   key == QStringLiteral("upright")) {
            if (!params.value(key).isBool()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 bool").arg(key);
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

std::unique_ptr<IDescriptorWorker> SurfDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[SURF] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<SurfDescriptorWorker>(params);
}

// ── KAZE ────────────────────────────────────────────────────────────────────

KazeDescriptorWorker::KazeDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("extended")))
        m_extended = params.value(QStringLiteral("extended")).toBool(false);
    if (params.contains(QStringLiteral("upright")))
        m_upright = params.value(QStringLiteral("upright")).toBool(false);
    if (params.contains(QStringLiteral("threshold")))
        m_threshold = params.value(QStringLiteral("threshold")).toDouble(0.001);
    if (params.contains(QStringLiteral("nOctaves")))
        m_nOctaves = params.value(QStringLiteral("nOctaves")).toInt(4);
    if (params.contains(QStringLiteral("nOctaveLayers")))
        m_nOctaveLayers = params.value(QStringLiteral("nOctaveLayers")).toInt(4);
    if (params.contains(QStringLiteral("diffusivity")))
        m_diffusivity = params.value(QStringLiteral("diffusivity")).toInt(1);
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(0);
}

FeatureResult KazeDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    auto kaze = cv::KAZE::create(m_extended, m_upright, m_threshold,
                                 m_nOctaves, m_nOctaveLayers,
                                 static_cast<cv::KAZE::DiffusivityType>(m_diffusivity));

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    kaze->detectAndCompute(image.imageGray, cv::noArray(),
                           r.keypoints, r.descriptors);

    // 限制特征点最大数量
    if (m_maxFeatures > 0 &&
        static_cast<int>(r.keypoints.size()) > m_maxFeatures) {
        r.keypoints.resize(m_maxFeatures);
        r.descriptors = r.descriptors.rowRange(0, m_maxFeatures).clone();
    }

    r.elapsedMs = 0.0;
    return r;
}

QList<ParamSchema> KazeDescriptorPlugin::paramSchema() const
{
    ParamSchema extended;
    extended.name = QStringLiteral("extended");
    extended.type = ParamSchema::Type::Bool;
    extended.defaultValue = false;
    extended.group = QStringLiteral("KAZE");
    extended.tooltip = QStringLiteral("是否使用扩展(128维)描述子");

    ParamSchema upright;
    upright.name = QStringLiteral("upright");
    upright.type = ParamSchema::Type::Bool;
    upright.defaultValue = false;
    upright.group = QStringLiteral("KAZE");
    upright.tooltip = QStringLiteral("是否使用无旋转 Upright 描述子");

    ParamSchema threshold;
    threshold.name = QStringLiteral("threshold");
    threshold.type = ParamSchema::Type::Double;
    threshold.defaultValue = 0.001;
    threshold.min = 0.0;
    threshold.group = QStringLiteral("KAZE");
    threshold.tooltip = QStringLiteral("检测响应阈值");

    ParamSchema nOctaves;
    nOctaves.name = QStringLiteral("nOctaves");
    nOctaves.type = ParamSchema::Type::Int;
    nOctaves.defaultValue = 4;
    nOctaves.min = 1;
    nOctaves.max = 16;
    nOctaves.group = QStringLiteral("KAZE");

    ParamSchema nOctaveLayers;
    nOctaveLayers.name = QStringLiteral("nOctaveLayers");
    nOctaveLayers.type = ParamSchema::Type::Int;
    nOctaveLayers.defaultValue = 4;
    nOctaveLayers.min = 1;
    nOctaveLayers.max = 16;
    nOctaveLayers.group = QStringLiteral("KAZE");

    ParamSchema diffusivity;
    diffusivity.name = QStringLiteral("diffusivity");
    diffusivity.type = ParamSchema::Type::Int;
    diffusivity.defaultValue = 1;
    diffusivity.min = 0;
    diffusivity.max = 3;
    diffusivity.group = QStringLiteral("KAZE");
    diffusivity.enumValues = { QStringLiteral("PM_G1"), QStringLiteral("PM_G2"),
                               QStringLiteral("WEICKERT"), QStringLiteral("CHARBONNIER") };
    diffusivity.tooltip = QStringLiteral("扩散类型：0=PM_G1 1=PM_G2 2=WEICKERT 3=CHARBONNIER");

    ParamSchema maxFeatures;
    maxFeatures.name = QStringLiteral("maxFeatures");
    maxFeatures.type = ParamSchema::Type::Int;
    maxFeatures.defaultValue = 0;
    maxFeatures.min = 0;
    maxFeatures.group = QStringLiteral("KAZE");
    maxFeatures.tooltip = QStringLiteral("最大特征点数量，0 表示不限制");

    return { extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity, maxFeatures };
}

bool KazeDescriptorPlugin::validateParams(const QJsonObject &params,
                                          QString *errorMsg) const
{
    const auto keys = params.keys();
    for (const auto &key : keys) {
        if (key == QStringLiteral("extended") ||
            key == QStringLiteral("upright")) {
            if (!params.value(key).isBool()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 bool").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("nOctaves") ||
                   key == QStringLiteral("nOctaveLayers") ||
                   key == QStringLiteral("diffusivity") ||
                   key == QStringLiteral("maxFeatures")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 int").arg(key);
                return false;
            }
        } else if (key == QStringLiteral("threshold")) {
            if (!params.value(key).isDouble()) {
                if (errorMsg)
                    *errorMsg = QStringLiteral("参数 %1 类型应为 double").arg(key);
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

std::unique_ptr<IDescriptorWorker> KazeDescriptorPlugin::createWorker(const QJsonObject &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning() << "[KAZE] 参数无效:" << err;
        return nullptr;
    }
    return std::make_unique<KazeDescriptorWorker>(params);
}
