#include "PatchMatchReconstructorV2.h"

#include "algorithm/dense/PatchMatchCost.h"
#include "algorithm/dense/PatchMatchFusion.h"
#include "algorithm/dense/PatchMatchPropagate.h"
#include "algorithm/dense/PatchMatchRefine.h"
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

QList<ParamSchema> PatchMatchReconstructorV2::paramSchema() const
{
    using sfm::makeBoolParam;
    using sfm::makeDoubleParam;
    using sfm::makeIntParam;
    // 默认值对齐 2026-10-03 全量验证录定配置
    return {
        makeDoubleParam(QStringLiteral("depthMin"), -1.0, -1.0, 1e6,
                        QStringLiteral("深度"), QStringLiteral("最小深度；<0 从稀疏点自动估计")),
        makeDoubleParam(QStringLiteral("depthMax"), -1.0, -1.0, 1e6,
                        QStringLiteral("深度"), QStringLiteral("最大深度；<0 从稀疏点自动估计")),
        makeIntParam(QStringLiteral("maxImageSize"), 1320, 0, 8000,
                     QStringLiteral("性能"), QStringLiteral("工作图像最大边长；0 = 不缩放（全分辨率）")),
        makeIntParam(QStringLiteral("windowRadius"), 5, 1, 20,
                     QStringLiteral("性能"), QStringLiteral("NCC 匹配窗口半径（半宽）")),
        makeIntParam(QStringLiteral("numSamples"), 15, 1, 100,
                     QStringLiteral("性能"), QStringLiteral("每像素随机采样数（初始化）")),
        makeIntParam(QStringLiteral("numIterations"), 5, 1, 50,
                     QStringLiteral("性能"), QStringLiteral("邻域传播迭代轮数（多尺度时为最粗层轮数）")),
        makeBoolParam(QStringLiteral("multiscale"), true,
                      QStringLiteral("性能"), QStringLiteral("多尺度金字塔 + JBU 上采样（提速并支撑高分辨率）")),
        makeBoolParam(QStringLiteral("planePrior"), true,
                      QStringLiteral("性能"),
                      QStringLiteral("平面先验（ACMP 受限后验：弱纹理区域更完整、重投影误差更低，耗时约 +12%）")),
        makeIntParam(QStringLiteral("costTopK"), 4, 1, 10,
                     QStringLiteral("代价"), QStringLiteral("多视图代价聚合：取最优 k 个源视图代价平均（抗遮挡/高光）")),
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
        makeBoolParam(QStringLiteral("geomConsistency"), true,
                      QStringLiteral("过滤"), QStringLiteral("几何一致性过滤（往返重投影剔除离群点）")),
        makeBoolParam(QStringLiteral("fusion"), true,
                      QStringLiteral("过滤"), QStringLiteral("深度融合：跨视图去重 + 置信度加权平均（降离群点、点云更紧致）")),
    };
}

void PatchMatchReconstructorV2::Options::loadFromParams(const QMap<QString, QVariant> &params)
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
    costTopK = params.value("costTopK", costTopK).toInt();
    filterMinNumConsistent = params.value("filterMinNumConsistent", filterMinNumConsistent).toInt();
    maxScales = params.value("maxScales", maxScales).toInt();
    refineIterations = params.value("refineIterations", refineIterations).toInt();
    multiscale = params.value("multiscale", multiscale).toBool();
    geomConsistency = params.value("geomConsistency", geomConsistency).toBool();
    fusion = params.value("fusion", fusion).toBool();
    fusionMaxRelDepthErr = params.value("fusionMaxRelDepthErr", fusionMaxRelDepthErr).toDouble();
    fusionMinSupport = params.value("fusionMinSupport", fusionMinSupport).toInt();
    planePrior = params.value("planePrior", planePrior).toBool();
    planePriorMaxCost = params.value("planePriorMaxCost", planePriorMaxCost).toFloat();
    planePriorStep = params.value("planePriorStep", planePriorStep).toInt();
    filter = params.value("filter", filter).toBool();
    allowMissingFiles = params.value("allowMissingFiles", allowMissingFiles).toBool();
    writeConsistencyGraph = params.value("writeConsistencyGraph", writeConsistencyGraph).toBool();
}

void PatchMatchReconstructorV2::setConfig(const QMap<QString, QVariant> &params)
{
    QString err;
    if (!validateParams(params, &err)) {
        qWarning("[PatchMatchV2] 参数无效，忽略本次 setConfig: %s", qPrintable(err));
        return;
    }
    m_options.loadFromParams(params);
}

DenseReconstructionOutput PatchMatchReconstructorV2::reconstruct(const DenseReconstructionInput &input)
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

    // 3.5 多视图几何一致性过滤（可选，需要全部视角深度图都已算完）
    // 当前门槛：既要开启 geomConsistency，也要开启 filter（保持 filter 作为总开关）
    if (m_options.geomConsistency && m_options.filter) {
        if (isStopRequested()) {
            output.errorMessage = QStringLiteral("已取消");
            output.denseCloud->clear();
            return output;
        }
        filterGeometricConsistency(input, results, pool);
    }

    // 4. 反投影生成密集点云（可选深度融合：跨视图去重 + 加权平均）
    if (isStopRequested()) {
        output.errorMessage = QStringLiteral("已取消");
        output.denseCloud->clear();
        return output;
    }
    if (m_options.fusion) {
        fuseDepthMaps(input, results, pool, *output.denseCloud);
    } else {
        for (const auto &result : results) {
            if (isStopRequested()) {
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
    }

    output.stereoResults = std::move(results);
    output.success = !output.denseCloud->empty();
    if (!output.success && output.errorMessage.isEmpty()) {
        output.errorMessage = QStringLiteral("未生成有效密集点云");
    }

    log(QString("[PatchMatch] 重建完成，输出点云点数: %1").arg(output.denseCloud->size()));
    return output;
}

std::vector<PatchMatchReconstructorV2::Problem> PatchMatchReconstructorV2::buildProblems(
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

std::vector<std::pair<double, double>> PatchMatchReconstructorV2::estimateDepthRanges(
    const DenseReconstructionInput &input) const
{
    return sfm::dense::estimateDepthRanges(input);
}

// ═══════════════════════════════════════════════════════════════
// PatchMatch 核心算法
// ═══════════════════════════════════════════════════════════════

float PatchMatchReconstructorV2::computePixelCost(int x, int y, float depth,
                                                 float nx, float ny, float nz,
                                                 const Context &ctx) const
{
    // F15c：热点代价委托给共享自由函数（与历史实现逐位一致）
    sfm::dense::PatchContextPod pod;
    pod.refGray = &ctx.refGray;
    pod.fx = ctx.fx;
    pod.fy = ctx.fy;
    pod.cx = ctx.cx;
    pod.cy = ctx.cy;
    pod.sources = ctx.sources.empty() ? nullptr : ctx.sources.data();
    pod.numSources = static_cast<int>(ctx.sources.size());
    pod.opt.windowRadius = m_options.windowRadius;
    pod.opt.sigmaSpatial = m_options.sigmaSpatial;
    pod.opt.sigmaColor = m_options.sigmaColor;
    pod.opt.costTopK = m_options.costTopK;
    return sfm::dense::computeNccPixelCost(x, y, depth, nx, ny, nz, pod);
}

void PatchMatchReconstructorV2::processPixel(int x, int y,
                                           cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                                           const Context &ctx,
                                           bool propagateDown,
                                           const sfm::pmv2::PlanarPrior *prior,
                                           std::mt19937 &rng) const
{
    // F15d：编排抽至 dense::processPixel；代价仍走本类 computePixelCost（→ NCC POD）
    sfm::dense::PixelCostFn costFn = [this, &ctx](int px, int py, float d, float nx, float ny,
                                                  float nz) {
        return computePixelCost(px, py, d, nx, ny, nz, ctx);
    };
    sfm::dense::processPixel(x, y, depthMap, normalMap, confidence, ctx.rows, ctx.cols, ctx.fx,
                             ctx.fy, ctx.cx, ctx.cy, ctx.depthMin, ctx.depthMax, propagateDown,
                             prior, rng, costFn);
}

cv::Mat PatchMatchReconstructorV2::runPatchMatch(const Context &ctx, cv::Mat &confidence,
                                               cv::Mat &normalMap,
                                               const cv::Mat &initDepth,
                                               const cv::Mat &initNormal,
                                               int numIters) const
{
    // F15e：init / 红黑迭代 / 代价过滤抽至 dense；本函数只组配置与日志
    const int rows = ctx.rows;
    const int cols = ctx.cols;
    const double depthMin = ctx.depthMin;
    const double depthMax = ctx.depthMax;

    cv::Mat depthMap(rows, cols, CV_32FC1);
    normalMap.create(rows, cols, CV_32FC3);
    confidence.create(rows, cols, CV_32FC1);

    const bool hasInit = !initDepth.empty() && !initNormal.empty() &&
                         initDepth.size() == depthMap.size() &&
                         initNormal.size() == normalMap.size();
    const int reused =
        sfm::dense::initDepthNormalMaps(depthMap, normalMap, depthMin, depthMax, initDepth,
                                        initNormal);
    if (hasInit) {
        log(QString("[PatchMatch] 初始化: 复用上采样结果 %1/%2 像素, 其余随机")
                .arg(reused)
                .arg(rows * cols));
    }

    log(QString("[PatchMatch] 初始化: %1x%2, 深度范围 [%3, %4], %5 源视图, %6 轮")
            .arg(cols)
            .arg(rows)
            .arg(depthMin, 0, 'f', 2)
            .arg(depthMax, 0, 'f', 2)
            .arg(ctx.sources.size())
            .arg(numIters));

    if (!ctx.pool) {
        log(QStringLiteral("[PatchMatch] 错误: 线程池未设置"));
        return depthMap;
    }

    sfm::dense::PatchMatchRunConfig cfg;
    cfg.numIters = numIters;
    cfg.wantPrior = m_options.planePrior && m_strategies.prior && m_strategies.prior->enabled() &&
                    numIters >= 2;
    cfg.planePriorStep = m_options.planePriorStep;
    cfg.planePriorMaxCost = m_options.planePriorMaxCost;
    cfg.fx = ctx.fx;
    cfg.fy = ctx.fy;
    cfg.cx = ctx.cx;
    cfg.cy = ctx.cy;
    cfg.depthMin = depthMin;
    cfg.depthMax = depthMax;

    sfm::dense::PixelCostFn costFn = [this, &ctx](int px, int py, float d, float nx, float ny,
                                                  float nz) {
        return computePixelCost(px, py, d, nx, ny, nz, ctx);
    };

    const bool ok = sfm::dense::runPatchMatchIterations(
        *ctx.pool, depthMap, normalMap, confidence, cfg, costFn,
        [this]() { return isStopRequested(); },
        [this](const sfm::dense::PlanarPriorEvent &ev) {
            if (ev.applied) {
                log(QString("[PatchMatch] 平面先验: %1 支撑点, 覆盖 %2/%3 像素 (%4%), %5 ms")
                        .arg(ev.supportCount)
                        .arg(ev.coverage)
                        .arg(ev.totalPixels)
                        .arg(100.0 * ev.coverage / std::max(1, ev.totalPixels), 0, 'f', 1)
                        .arg(ev.elapsedMs));
            } else {
                log(QString("[PatchMatch] 平面先验: 支撑点不足(%1)，本层退回纯光度")
                        .arg(ev.supportCount));
            }
        },
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

void PatchMatchReconstructorV2::jointBilateralUpsample(
    const cv::Mat &depthLow, const cv::Mat &normalLow,
    const cv::Mat &grayHigh,
    cv::Mat &depthHigh, cv::Mat &normalHigh) const
{
    sfm::dense::jointBilateralUpsample(depthLow, normalLow, grayHigh, depthHigh, normalHigh);
}

void PatchMatchReconstructorV2::filterGeometricConsistency(
    const DenseReconstructionInput &input,
    std::vector<StereoPairResult> &results,
    sfm::ThreadPool &pool) const
{
    sfm::dense::GeomConsistencyOptions opt;
    opt.maxReprojPx = m_options.filterGeomConsistencyMaxCost;
    opt.minConsistent = m_options.filterMinNumConsistent;

    const auto stats =
        sfm::dense::filterGeometricConsistency(input, results, pool, opt);
    if (stats.skippedTooFewViews) {
        log(QStringLiteral("[PatchMatch] 视图数不足，跳过几何一致性过滤"));
        return;
    }

    log(QString("[PatchMatch] 几何一致性过滤: 阈值=%1 px, 最少一致视图=%2, 保留 %3 / %4 (%5%)")
            .arg(opt.maxReprojPx, 0, 'f', 2)
            .arg(std::max(1, opt.minConsistent))
            .arg(stats.totalKept)
            .arg(stats.totalValid)
            .arg(stats.totalValid > 0 ? 100.0 * stats.totalKept / stats.totalValid : 0.0, 0,
                 'f', 1));
}

StereoPairResult PatchMatchReconstructorV2::solveProblem(
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
        sfm::dense::MaxImageSizePolicy::UnlimitedWhenNonPositive, loadByIdx);
    if (!prep.ok)
        return result;

    Context ctx;
    ctx.refGray = prep.refGray;
    ctx.K_ref = prep.K_work;
    ctx.fx = prep.K_work.at<double>(0, 0);
    ctx.fy = prep.K_work.at<double>(1, 1);
    ctx.cx = prep.K_work.at<double>(0, 2);
    ctx.cy = prep.K_work.at<double>(1, 2);
    ctx.rows = prep.refGray.rows;
    ctx.cols = prep.refGray.cols;
    ctx.refImageIdx = prep.refImageIdx;
    ctx.pool = &pool;
    ctx.depthMin = prep.depthMin;
    ctx.depthMax = prep.depthMax;
    ctx.sources = sfm::dense::buildPodSourceViews(input, prep.refImageIdx, problem.srcImageIdxs,
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

    // ── 运行 PatchMatch（单尺度或多尺度 + JBU）─────────────────────────
    cv::Mat confidence;
    cv::Mat depth;
    cv::Mat normalMap;

    const bool msEnabled =
        m_options.multiscale && m_strategies.refine && m_strategies.refine->enabled();
    const int numScales =
        sfm::dense::decideNumScales(ctx.cols, ctx.rows, m_options.maxScales, msEnabled);

    auto levelFromContext = [](const Context &c) {
        sfm::dense::PatchMatchLevelCtx lv;
        lv.refGray = c.refGray;
        lv.K_ref = c.K_ref;
        lv.fx = c.fx;
        lv.fy = c.fy;
        lv.cx = c.cx;
        lv.cy = c.cy;
        lv.sources = c.sources;
        lv.depthMin = c.depthMin;
        lv.depthMax = c.depthMax;
        lv.rows = c.rows;
        lv.cols = c.cols;
        lv.refImageIdx = c.refImageIdx;
        lv.pool = c.pool;
        return lv;
    };
    auto contextFromLevel = [](const sfm::dense::PatchMatchLevelCtx &lv) {
        Context c;
        c.refGray = lv.refGray;
        c.K_ref = lv.K_ref;
        c.fx = lv.fx;
        c.fy = lv.fy;
        c.cx = lv.cx;
        c.cy = lv.cy;
        c.sources = lv.sources;
        c.depthMin = lv.depthMin;
        c.depthMax = lv.depthMax;
        c.rows = lv.rows;
        c.cols = lv.cols;
        c.refImageIdx = lv.refImageIdx;
        c.pool = lv.pool;
        return c;
    };

    if (numScales <= 1) {
        depth = runPatchMatch(ctx, confidence, normalMap, cv::Mat(), cv::Mat(),
                              m_options.numIterations);
    } else {
        auto pyr = sfm::dense::buildPyramid(levelFromContext(ctx), numScales);
        log(QString("[PatchMatch] 多尺度: %1 层 (最细 %2x%3 -> 最粗 %4x%5)")
                .arg(numScales)
                .arg(pyr[0].cols)
                .arg(pyr[0].rows)
                .arg(pyr[numScales - 1].cols)
                .arg(pyr[numScales - 1].rows));

        const bool ok = sfm::dense::runCoarseToFine(
            pyr, m_options.numIterations, m_options.refineIterations,
            [this]() { return isStopRequested(); },
            [this](int step, int nScales, int cols, int rows, int iters) {
                log(QString("[PatchMatch] ── 层级 %1/%2 (%3x%4), %5 轮 ──")
                        .arg(step)
                        .arg(nScales)
                        .arg(cols)
                        .arg(rows)
                        .arg(iters));
            },
            [this, &contextFromLevel](sfm::dense::PatchMatchLevelCtx &level, cv::Mat &conf,
                                      cv::Mat &normals, const cv::Mat &initD,
                                      const cv::Mat &initN, int iters) {
                Context c = contextFromLevel(level);
                return runPatchMatch(c, conf, normals, initD, initN, iters);
            },
            depth, confidence, normalMap);
        if (!ok)
            depth.release();
    }

    // depth 与 confidence 必须一起上采样到全分辨率（历史曾只缩放 depth 导致错位）
    if (prep.scale < 1.0)
        sfm::dense::upscaleToFullResolution(depth, confidence, prep.refColor.size(), true);

    const int validCount = sfm::dense::countValidDepthPixels(depth);
    log(QString("[PatchMatch] 深度图统计: %1x%2, 有效像素 %3 (%4%)")
            .arg(depth.cols)
            .arg(depth.rows)
            .arg(validCount)
            .arg(depth.empty() ? 0.0 : 100.0 * validCount / (depth.rows * depth.cols), 0, 'f',
                 1));

    result.depth = depth;
    result.confidence = confidence;
    result.rectifiedImage1 = prep.refColor;
    result.K_rect = prep.K_full.clone();
    result.R1_rect = cv::Mat::eye(3, 3, CV_64FC1);
    return result;
}

void PatchMatchReconstructorV2::fuseDepthMaps(
    const DenseReconstructionInput &input,
    const std::vector<StereoPairResult> &results,
    sfm::ThreadPool &pool,
    pcl::PointCloud<pcl::PointXYZRGB> &cloud) const
{
    const int numRes = static_cast<int>(results.size());
    if (numRes == 0)
        return;

    std::vector<cv::Mat> images(static_cast<size_t>(numRes));
    for (int i = 0; i < numRes; ++i) {
        const int imgIdx = results[static_cast<size_t>(i)].imageId1;
        if (imgIdx < 0 || imgIdx >= input.imageFiles.size())
            continue;
        if (results[static_cast<size_t>(i)].depth.empty())
            continue;
        images[static_cast<size_t>(i)] = loadImage(input.imageFiles[imgIdx]);
    }

    sfm::dense::FusionOptions opt;
    opt.relErrMax = m_options.fusionMaxRelDepthErr;
    opt.minSupport = m_options.fusionMinSupport;

    const auto stats =
        sfm::dense::fuseDepthMaps(input, results, images, pool, opt, cloud);
    log(QString("[PatchMatch] 深度融合: %1 点 (被合并跳过 %2, 支持不足剔除 %3, "
                "relErr<%4, minSupport=%5, %6 s)")
            .arg(stats.numEmitted)
            .arg(stats.numConsumed)
            .arg(stats.numLowSupport)
            .arg(opt.relErrMax)
            .arg(std::max(1, opt.minSupport))
            .arg(stats.seconds, 0, 'f', 1));
}

