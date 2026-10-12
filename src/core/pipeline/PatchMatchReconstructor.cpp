#include "PatchMatchReconstructor.h"

#include "algorithm/dense/PatchMatchCost.h"
#include "algorithm/dense/PatchMatchPropagate.h"
#include "algorithm/dense/PatchMatchShared.h"
#include "DenseReconstructionFactory.h"
#include "algorithm/ThreadPool.h"
#include "plugin/ParamSchemaUtils.h"

#include <QtGlobal>

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace {

inline cv::Mat loadImage(const QString &path)
{
    return cv::imread(path.toStdString(), cv::IMREAD_COLOR);
}

} // namespace

// 注：PatchMatch 由 DenseReconstructionFactory::registerBuiltinMethods() 显式注册
//（静态库链接时，此处的静态注册对象可能不被拉入）。

QList<ParamSchema> PatchMatchReconstructor::paramSchema() const
{
    using sfm::makeBoolParam;
    using sfm::makeDoubleParam;
    using sfm::makeIntParam;
    return {
        makeDoubleParam(QStringLiteral("depthMin"), -1.0, -1.0, 1e6,
                        QStringLiteral("深度"), QStringLiteral("最小深度；<0 从稀疏点自动估计")),
        makeDoubleParam(QStringLiteral("depthMax"), -1.0, -1.0, 1e6,
                        QStringLiteral("深度"), QStringLiteral("最大深度；<0 从稀疏点自动估计")),
        makeIntParam(QStringLiteral("maxImageSize"), 1000, 64, 5000,
                     QStringLiteral("性能"), QStringLiteral("CPU 实现的工作图像最大边长（缩小可显著提速）")),
        makeIntParam(QStringLiteral("windowRadius"), 5, 1, 20,
                     QStringLiteral("性能"), QStringLiteral("NCC 匹配窗口半径（半宽）")),
        makeIntParam(QStringLiteral("numSamples"), 15, 1, 100,
                     QStringLiteral("性能"), QStringLiteral("每像素随机采样数（初始化）")),
        makeIntParam(QStringLiteral("numIterations"), 5, 1, 50,
                     QStringLiteral("性能"), QStringLiteral("邻域传播迭代轮数")),
        makeDoubleParam(QStringLiteral("nccSigma"), 0.6, 0.1, 5.0,
                        QStringLiteral("代价"), QStringLiteral("NCC 似然函数扩散系数")),
        makeDoubleParam(QStringLiteral("sigmaColor"), 51.0, 1.0, 255.0,
                        QStringLiteral("代价"), QStringLiteral("颜色权重 sigma（uint8 图像）")),
        makeDoubleParam(QStringLiteral("minTriangulationAngle"), 0.5, 0.1, 10.0,
                        QStringLiteral("代价"), QStringLiteral("选源图像最小光轴夹角（度）")),
        makeDoubleParam(QStringLiteral("filterMinNcc"), 0.1, 0.0, 1.0,
                        QStringLiteral("过滤"), QStringLiteral("过滤最小 NCC 阈值（代价上限）")),
        makeDoubleParam(QStringLiteral("filterMinTriangulationAngle"), 3.0, 0.0, 30.0,
                        QStringLiteral("过滤"), QStringLiteral("过滤最小三角化角度（度）")),
        makeBoolParam(QStringLiteral("filter"), true,
                      QStringLiteral("过滤"), QStringLiteral("启用置信度过滤")),
        makeBoolParam(QStringLiteral("geomConsistency"), false,
                      QStringLiteral("过滤"), QStringLiteral("几何一致性校验（当前保留未实现）")),
    };
}

void PatchMatchReconstructor::Options::loadFromParams(const QMap<QString, QVariant> &params)
{
    depthMin = params.value("depthMin", depthMin).toDouble();
    depthMax = params.value("depthMax", depthMax).toDouble();
    sigmaSpatial = params.value("sigmaSpatial", sigmaSpatial).toDouble();
    sigmaColor = params.value("sigmaColor", sigmaColor).toDouble();
    nccSigma = params.value("nccSigma", nccSigma).toDouble();
    minTriangulationAngle = params.value("minTriangulationAngle", minTriangulationAngle).toDouble();
    incidentAngleSigma = params.value("incidentAngleSigma", incidentAngleSigma).toDouble();
    geomConsistencyRegularizer = params.value("geomConsistencyRegularizer", geomConsistencyRegularizer).toDouble();
    geomConsistencyMaxCost = params.value("geomConsistencyMaxCost", geomConsistencyMaxCost).toDouble();
    filterMinNcc = params.value("filterMinNcc", filterMinNcc).toDouble();
    filterMinTriangulationAngle = params.value("filterMinTriangulationAngle", filterMinTriangulationAngle).toDouble();
    filterGeomConsistencyMaxCost = params.value("filterGeomConsistencyMaxCost", filterGeomConsistencyMaxCost).toDouble();
    cacheSize = params.value("cacheSize", cacheSize).toDouble();
    gpuIndex = params.value("gpuIndex", gpuIndex).toString();
    maxImageSize = params.value("maxImageSize", maxImageSize).toInt();
    windowRadius = params.value("windowRadius", windowRadius).toInt();
    windowStep = params.value("windowStep", windowStep).toInt();
    numSamples = params.value("numSamples", numSamples).toInt();
    numIterations = params.value("numIterations", numIterations).toInt();
    filterMinNumConsistent = params.value("filterMinNumConsistent", filterMinNumConsistent).toInt();
    geomConsistency = params.value("geomConsistency", geomConsistency).toBool();
    filter = params.value("filter", filter).toBool();
    allowMissingFiles = params.value("allowMissingFiles", allowMissingFiles).toBool();
    writeConsistencyGraph = params.value("writeConsistencyGraph", writeConsistencyGraph).toBool();
}

void PatchMatchReconstructor::setConfig(const QMap<QString, QVariant> &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning("[PatchMatch] 参数无效，忽略本次 setConfig: %s", qPrintable(err));
        return;
    }
    m_options.loadFromParams(params);
}

DenseReconstructionOutput PatchMatchReconstructor::reconstruct(const DenseReconstructionInput &input)
{
    DenseReconstructionOutput output;
    output.denseCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (input.cameraPoses.empty() || input.imageFiles.isEmpty()) {
        output.errorMessage = QStringLiteral("输入数据为空");
        return output;
    }
    if (isStopRequested()) {
        output.errorMessage = QStringLiteral("已取消");
        return output;
    }

    log(QStringLiteral("[PatchMatch] 开始多视图密集重建 (Homography warping + NCC)"));

    // 线程池：使用进程级全局共享池（与稀疏重建等其他阶段共用，避免线程超订）
    sfm::ThreadPool &pool = sfm::globalThreadPool();
    log(QString("[PatchMatch] 使用全局共享线程池: %1 线程, 红黑网格")
            .arg(pool.thread_count()));

    // 1. 构建 PatchMatch 问题：为每张参考图像选择源视图
    std::vector<Problem> problems = buildProblems(input);
    log(QString("[PatchMatch] 生成 %1 个参考视角问题").arg(problems.size()));
    if (problems.empty()) {
        output.errorMessage = QStringLiteral("没有有效的 PatchMatch 问题");
        return output;
    }

    // 2. 估计深度范围
    std::vector<std::pair<double, double>> depthRanges = estimateDepthRanges(input);

    // 3. 逐个求解参考视角深度图
    std::vector<StereoPairResult> results;
    results.reserve(problems.size());
    for (size_t i = 0; i < problems.size(); ++i) {
        if (isStopRequested()) {
            log(QStringLiteral("[PatchMatch] 收到停止请求，中止密集重建"));
            output.errorMessage = QStringLiteral("已取消");
            output.denseCloud->clear();
            return output;
        }
        const auto &problem = problems[i];
        log(QString("[PatchMatch] 处理参考视角 %1 / %2 (image %3, %4 个源视图)")
                .arg(i + 1).arg(problems.size())
                .arg(problem.refImageIdx).arg(problem.srcImageIdxs.size()));

        StereoPairResult result = solveProblem(input, problem, depthRanges[problem.refImageIdx], pool);
        if (isStopRequested()) {
            log(QStringLiteral("[PatchMatch] 收到停止请求，中止密集重建"));
            output.errorMessage = QStringLiteral("已取消");
            output.denseCloud->clear();
            return output;
        }
        if (!result.depth.empty()) {
            results.push_back(std::move(result));
        }
    }

    log(QString("[PatchMatch] 有效深度图数量: %1").arg(results.size()));
    if (results.empty()) {
        output.errorMessage = QStringLiteral("未生成有效深度图");
        return output;
    }

    // 4. 反投影生成密集点云
    for (const auto &result : results) {
        if (isStopRequested()) {
            log(QStringLiteral("[PatchMatch] 收到停止请求，中止密集重建"));
            output.errorMessage = QStringLiteral("已取消");
            output.denseCloud->clear();
            return output;
        }
        if (result.depth.empty()) continue;

        const int refIdx = result.imageId1;
        if (refIdx < 0 || refIdx >= input.imageFiles.size()) continue;

        cv::Mat refImage = loadImage(input.imageFiles[refIdx]);
        sfm::dense::backprojectDepth(input, refIdx, result.depth, refImage, *output.denseCloud);
    }

    output.stereoResults = std::move(results);
    output.success = !output.denseCloud->empty();
    if (!output.success && output.errorMessage.isEmpty()) {
        output.errorMessage = QStringLiteral("未生成有效密集点云");
    }

    log(QString("[PatchMatch] 重建完成，输出点云点数: %1").arg(output.denseCloud->size()));
    return output;
}

std::vector<PatchMatchReconstructor::Problem> PatchMatchReconstructor::buildProblems(
    const DenseReconstructionInput &input) const
{
    const auto shared = sfm::dense::buildProblems(input);
    std::vector<Problem> problems;
    problems.reserve(shared.size());
    for (const auto &p : shared) {
        Problem q;
        q.refImageIdx = p.refImageIdx;
        q.srcImageIdxs = p.srcImageIdxs;
        problems.push_back(std::move(q));
    }
    return problems;
}

std::vector<std::pair<double, double>> PatchMatchReconstructor::estimateDepthRanges(
    const DenseReconstructionInput &input) const
{
    return sfm::dense::estimateDepthRanges(input);
}

// ═══════════════════════════════════════════════════════════════
// PatchMatch 核心算法
// ═══════════════════════════════════════════════════════════════

float PatchMatchReconstructor::computePixelCost(int x, int y, float depth,
                                                 float nx, float ny, float nz,
                                                 const Context &ctx) const
{
    // F15i：历史 Mat NCC 已抽至 dense::computeNccPixelCostV1
    const cv::Mat &Kref = ctx.K_ref;
    sfm::dense::PatchContext pod;
    pod.refGray = &ctx.refGray;
    pod.fx = Kref.at<double>(0, 0);
    pod.fy = Kref.at<double>(1, 1);
    pod.cx = Kref.at<double>(0, 2);
    pod.cy = Kref.at<double>(1, 2);
    pod.sources = ctx.sources.empty() ? nullptr : ctx.sources.data();
    pod.numSources = static_cast<int>(ctx.sources.size());
    pod.windowRadius = m_options.windowRadius;
    pod.sigmaSpatial = m_options.sigmaSpatial;
    pod.sigmaColor = m_options.sigmaColor;
    pod.costTopK = 1;
    return sfm::dense::computeNccPixelCostV1(x, y, depth, nx, ny, nz, pod);
}

void PatchMatchReconstructor::processPixel(int x, int y,
                                           cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                                           const Context &ctx,
                                           bool propagateDown,
                                           std::mt19937 &rng) const
{
    // F15h：编排抽至 dense::processPixel（prior=nullptr，纯光度，与历史 V1 一致）
    const cv::Mat &Kref = ctx.K_ref;
    const double fx = Kref.at<double>(0, 0);
    const double fy = Kref.at<double>(1, 1);
    const double cx = Kref.at<double>(0, 2);
    const double cy = Kref.at<double>(1, 2);

    sfm::dense::PixelCostFn costFn = [this, &ctx](int px, int py, float d, float nx, float ny,
                                                  float nz) {
        return computePixelCost(px, py, d, nx, ny, nz, ctx);
    };
    sfm::dense::processPixel(x, y, depthMap, normalMap, confidence, ctx.rows, ctx.cols, fx, fy, cx,
                             cy, ctx.depthMin, ctx.depthMax, propagateDown, nullptr, rng, costFn);
}

cv::Mat PatchMatchReconstructor::runPatchMatch(const Context &ctx, cv::Mat &confidence) const
{
    // F15h：init / 红黑迭代 / 代价过滤抽至 dense；本函数只组配置与日志
    const int rows = ctx.rows;
    const int cols = ctx.cols;
    const double depthMin = ctx.depthMin;
    const double depthMax = ctx.depthMax;

    cv::Mat depthMap(rows, cols, CV_32FC1);
    cv::Mat normalMap(rows, cols, CV_32FC3);
    confidence.create(rows, cols, CV_32FC1);

    const cv::Mat &Kref = ctx.K_ref;
    const double fx = Kref.at<double>(0, 0);
    const double fy = Kref.at<double>(1, 1);
    const double cx = Kref.at<double>(0, 2);
    const double cy = Kref.at<double>(1, 2);

    (void)sfm::dense::initDepthNormalMaps(depthMap, normalMap, depthMin, depthMax, cv::Mat(),
                                          cv::Mat());

    log(QString("[PatchMatch] 初始化: %1x%2, 深度范围 [%3, %4], %5 源视图")
            .arg(cols)
            .arg(rows)
            .arg(depthMin, 0, 'f', 2)
            .arg(depthMax, 0, 'f', 2)
            .arg(ctx.sources.size()));

    if (!ctx.pool) {
        log(QStringLiteral("[PatchMatch] 错误: 线程池未设置"));
        return depthMap;
    }

    sfm::dense::PatchMatchRunConfig cfg;
    cfg.numIters = m_options.numIterations;
    cfg.wantPrior = m_strategies.prior && m_strategies.prior->enabled();
    cfg.fx = fx;
    cfg.fy = fy;
    cfg.cx = cx;
    cfg.cy = cy;
    cfg.depthMin = depthMin;
    cfg.depthMax = depthMax;

    sfm::dense::PixelCostFn costFn = [this, &ctx](int px, int py, float d, float nx, float ny,
                                                  float nz) {
        return computePixelCost(px, py, d, nx, ny, nz, ctx);
    };

    const bool ok = sfm::dense::runPatchMatchIterations(
        *ctx.pool, depthMap, normalMap, confidence, cfg, costFn,
        [this]() { return isStopRequested(); },
        [](const sfm::dense::PlanarPriorEvent &) {},
        [this](const sfm::dense::PatchMatchIterDiag &d) {
            log(QString("[PatchMatch] 迭代 %1/%2 完成 (%3 ms, 有效=%4%, 深度[%5,%6] 均值=%7, "
                        "均代价=%8)")
                    .arg(d.iter)
                    .arg(d.numIters)
                    .arg(d.elapsedMs)
                    .arg(100.0 * d.validCnt / std::max(1, d.totalPixels), 0, 'f', 1)
                    .arg(d.minD, 0, 'f', 2)
                    .arg(d.maxD, 0, 'f', 2)
                    .arg(d.meanDepth, 0, 'f', 2)
                    .arg(d.meanCost, 0, 'f', 4));
        });

    if (!ok) {
        log(QStringLiteral("[PatchMatch] 迭代中收到停止请求，提前结束"));
        return depthMap;
    }

    const float costThreshold = 1.0f - float(m_options.filterMinNcc);
    const int filteredCount =
        sfm::dense::filterDepthByCost(depthMap, confidence, costThreshold);
    log(QString("[PatchMatch] 代价过滤: 阈值=%1, 过滤 %2 像素 (%3%)")
            .arg(costThreshold, 0, 'f', 3)
            .arg(filteredCount)
            .arg(100.0 * filteredCount / (rows * cols), 0, 'f', 1));

    return depthMap;
}

StereoPairResult PatchMatchReconstructor::solveProblem(
    const DenseReconstructionInput &input,
    const Problem &problem,
    const std::pair<double, double> &depthRange,
    sfm::ThreadPool &pool) const
{
    StereoPairResult result;
    result.imageId1 = problem.refImageIdx;
    result.imageId2 = -1;

    auto loadByIdx = [&](int idx) -> cv::Mat {
        if (idx < 0 || idx >= input.imageFiles.size())
            return {};
        return loadImage(input.imageFiles[idx]);
    };

    const auto prep = sfm::dense::prepareReferenceView(
        input, problem.refImageIdx, depthRange, m_options.maxImageSize,
        sfm::dense::MaxImageSizePolicy::CapAt1000WhenNonPositive, loadByIdx);
    if (!prep.ok)
        return result;

    Context ctx;
    ctx.refGray = prep.refGray;
    ctx.K_ref = prep.K_work;
    ctx.rows = prep.refGray.rows;
    ctx.cols = prep.refGray.cols;
    ctx.refImageIdx = prep.refImageIdx;
    ctx.pool = &pool;
    ctx.depthMin = prep.depthMin;
    ctx.depthMax = prep.depthMax;
    ctx.sources = sfm::dense::buildMatSourceViews(input, prep.refImageIdx, problem.srcImageIdxs,
                                                  prep.scale, loadByIdx);

    if (ctx.sources.empty()) {
        log("[PatchMatch] 无有效源视图，跳过");
        return result;
    }

    log(QString("[PatchMatch] 参考 image %1: 工作尺寸 %2x%3, %4 源视图, 深度 [%5, %6]")
            .arg(prep.refImageIdx)
            .arg(ctx.cols)
            .arg(ctx.rows)
            .arg(ctx.sources.size())
            .arg(ctx.depthMin, 0, 'f', 2)
            .arg(ctx.depthMax, 0, 'f', 2));

    cv::Mat confidence;
    cv::Mat depth = runPatchMatch(ctx, confidence);
    if (prep.scale < 1.0)
        sfm::dense::upscaleToFullResolution(depth, confidence, prep.refColor.size(), false);

    const int validCount = sfm::dense::countValidDepthPixels(depth);
    log(QString("[PatchMatch] 深度图统计: %1x%2, 有效像素 %3 (%4%)")
            .arg(depth.cols)
            .arg(depth.rows)
            .arg(validCount)
            .arg(depth.empty() ? 0.0 : 100.0 * validCount / (depth.rows * depth.cols), 0, 'f', 1));

    result.depth = depth;
    result.confidence = confidence;
    result.rectifiedImage1 = prep.refColor;
    result.K_rect = prep.K_full.clone();
    result.R1_rect = cv::Mat::eye(3, 3, CV_64FC1);
    return result;
}

