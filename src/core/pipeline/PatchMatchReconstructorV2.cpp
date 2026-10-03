#include "PatchMatchReconstructorV2.h"
#include "DenseReconstructionFactory.h"
#include "algorithm/ThreadPool.h"

#include <QElapsedTimer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <mutex>
#include <random>

namespace {

inline cv::Mat loadImage(const QString &path)
{
    return cv::imread(path.toStdString(), cv::IMREAD_COLOR);
}

inline double angleBetweenDirections(const cv::Mat &R1, const cv::Mat &R2)
{
    // 光轴为 R 的第三行（相机坐标系 z 轴在世界坐标系下的方向）
    cv::Mat z1 = (cv::Mat_<double>(3, 1) << R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2));
    cv::Mat z2 = (cv::Mat_<double>(3, 1) << R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2));
    double dot = z1.dot(z2);
    dot = std::max(-1.0, std::min(1.0, dot));
    return std::acos(dot) * 180.0 / CV_PI;
}

} // namespace

// 注：PatchMatch 由 DenseReconstructionFactory::registerBuiltinMethods() 显式注册
//（静态库链接时，此处的静态注册对象可能不被拉入）。

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
        const auto &problem = problems[i];
        log(QString("[PatchMatch] 处理参考视角 %1 / %2 (image %3, %4 个源视图)")
                .arg(i + 1).arg(problems.size())
                .arg(problem.refImageIdx).arg(problem.srcImageIdxs.size()));

        StereoPairResult result = solveProblem(input, problem, depthRanges[problem.refImageIdx], pool);
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
        filterGeometricConsistency(input, results, pool);
    }

    // 4. 反投影生成密集点云（可选深度融合：跨视图去重 + 加权平均）
    if (m_options.fusion) {
        fuseDepthMaps(input, results, pool, *output.denseCloud);
    } else {
        for (const auto &result : results) {
            if (result.depth.empty()) continue;

            const int refIdx = result.imageId1;
            if (refIdx < 0 || refIdx >= input.imageFiles.size()) continue;

            cv::Mat refImage = loadImage(input.imageFiles[refIdx]);
            backprojectDepth(input, refIdx, result.depth, refImage, *output.denseCloud);
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
    const int numImages = static_cast<int>(input.imageFiles.size());
    ViewGraph viewGraph = ViewGraph::buildFromPairs(input.matchPairs, numImages);

    std::vector<Problem> problems;
    problems.reserve(numImages);

    for (int refIdx = 0; refIdx < numImages; ++refIdx) {
        if (refIdx >= static_cast<int>(input.cameraPoses.size())) continue;
        if (!input.cameraPoses[refIdx].registered) continue;

        Problem problem;
        problem.refImageIdx = refIdx;

        // 收集所有邻居，按匹配点数排序
        std::vector<std::pair<int, int>> candidates; // (srcIdx, matchCount)
        for (int edgeIdx : viewGraph.adj[refIdx]) {
            const auto &edge = viewGraph.edges[edgeIdx];
            int nb = (edge.nodeA == refIdx) ? edge.nodeB : edge.nodeA;
            if (nb < 0 || nb >= numImages) continue;
            if (!input.cameraPoses[nb].registered) continue;
            candidates.emplace_back(nb, edge.matchCount);
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        // 按三角化角度过滤，最多保留 10 个源视图
        const auto &refPose = input.cameraPoses[refIdx];
        for (const auto &[srcIdx, matchCount] : candidates) {
            if (problem.srcImageIdxs.size() >= 10) break;
            if (matchCount < 5) continue;

            const auto &srcPose = input.cameraPoses[srcIdx];
            // 光轴夹角过大时过滤（NCC窗口内容差异太大）
            double angle = angleBetweenDirections(refPose.R, srcPose.R);
            if (angle > 60.0) continue;

            problem.srcImageIdxs.push_back(srcIdx);
        }

        if (!problem.srcImageIdxs.empty()) {
            problems.push_back(std::move(problem));
        }
    }

    return problems;
}

std::vector<std::pair<double, double>> PatchMatchReconstructorV2::estimateDepthRanges(
    const DenseReconstructionInput &input) const
{
    const int numImages = static_cast<int>(input.imageFiles.size());
    std::vector<std::pair<double, double>> ranges(numImages, {-1.0, -1.0});

    // 通过三角化稀疏匹配点估计每张图像的深度分布
    std::vector<std::vector<double>> depthsPerImage(numImages);

    for (const auto &mp : input.matchPairs) {
        if (mp.imageId1 < 0 || mp.imageId1 >= numImages ||
            mp.imageId2 < 0 || mp.imageId2 >= numImages) continue;
        if (mp.points1.size() != mp.points2.size() || mp.points1.empty()) continue;

        const auto &pose1 = input.cameraPoses[mp.imageId1];
        const auto &pose2 = input.cameraPoses[mp.imageId2];
        if (!pose1.registered || !pose2.registered) continue;

        const cv::Mat &K1 = pose1.K.empty() ? input.cameraMatrix : pose1.K;
        const cv::Mat &K2 = pose2.K.empty() ? input.cameraMatrix : pose2.K;
        if (K1.empty() || K2.empty()) continue;

        // 构建投影矩阵 P = K · [R | -R·C] = K · [R | t]
        // 其中 t = -R·C, C = -R^T·t (相机中心)
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        {
            cv::Mat KR1 = K1 * pose1.R;
            KR1.copyTo(P1(cv::Rect(0, 0, 3, 3)));
            cv::Mat Kt1 = K1 * pose1.t;
            Kt1.copyTo(P1(cv::Rect(3, 0, 1, 3)));
        }
        cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
        {
            cv::Mat KR2 = K2 * pose2.R;
            KR2.copyTo(P2(cv::Rect(0, 0, 3, 3)));
            cv::Mat Kt2 = K2 * pose2.t;
            Kt2.copyTo(P2(cv::Rect(3, 0, 1, 3)));
        }

        // 相机中心 C = -R^T · t
        cv::Mat C1 = -pose1.R.t() * pose1.t;
        cv::Mat C2 = -pose2.R.t() * pose2.t;

        const size_t n = std::min(mp.points1.size(), size_t(500));
        for (size_t i = 0; i < n; ++i) {
            const cv::Point2d &p1 = mp.points1[i];
            const cv::Point2d &p2 = mp.points2[i];

            // 线性三角化（DLT）
            // 注意：MatExpr 没有 copyTo，需先构造为 Mat 再 copyTo 到目标行
            cv::Mat A(4, 4, CV_64F);
            cv::Mat(p1.x * P1.row(2) - P1.row(0)).reshape(1, 1).copyTo(A.row(0));
            cv::Mat(p1.y * P1.row(2) - P1.row(1)).reshape(1, 1).copyTo(A.row(1));
            cv::Mat(p2.x * P2.row(2) - P2.row(0)).reshape(1, 1).copyTo(A.row(2));
            cv::Mat(p2.y * P2.row(2) - P2.row(1)).reshape(1, 1).copyTo(A.row(3));

            cv::Mat S, U, Vt;
            cv::SVD::compute(A, S, U, Vt);

            cv::Mat Xh = Vt.row(3).t();
            if (std::abs(Xh.at<double>(3)) < 1e-9) continue;
            cv::Mat X = Xh / Xh.at<double>(3);  // 齐次坐标 → 笛卡尔

            // 转换到各相机坐标系，取 Z 作为深度
            cv::Mat Xw = X(cv::Rect(0, 0, 1, 3)).clone();
            cv::Mat Xc1 = pose1.R * Xw + pose1.t;
            double depth1 = Xc1.at<double>(2);
            if (depth1 > 0 && std::isfinite(depth1)) {
                depthsPerImage[mp.imageId1].push_back(depth1);
            }

            cv::Mat Xc2 = pose2.R * Xw + pose2.t;
            double depth2 = Xc2.at<double>(2);
            if (depth2 > 0 && std::isfinite(depth2)) {
                depthsPerImage[mp.imageId2].push_back(depth2);
            }
        }
    }

    // 计算每张图像的深度分位数
    for (int i = 0; i < numImages; ++i) {
        auto &depths = depthsPerImage[i];
        if (depths.size() < 5) continue;

        std::sort(depths.begin(), depths.end());
        size_t p5 = depths.size() * 5 / 100;
        size_t p95 = depths.size() * 95 / 100;
        if (p95 <= p5) p95 = p5 + 1;

        double dMin = depths[p5];
        double dMax = depths[p95];

        double margin = (dMax - dMin) * 0.2;
        ranges[i] = {std::max(0.01, dMin - margin), dMax + margin};
    }

    // 限制 depthMax/depthMin 比例，避免过宽范围导致 PatchMatch 无法收敛
    // 当 ratio 过大时（通常因稀疏匹配外点），用稀疏点深度的中位数重新计算
    // 经验值：PatchMatch 在 ratio <= 30 时收敛良好，>100 时几乎无法收敛
    const double MAX_DEPTH_RATIO = 30.0;
    int clampedCount = 0;
    for (int i = 0; i < numImages; ++i) {
        auto &r = ranges[i];
        if (r.first <= 0) continue;
        double ratio = r.second / r.first;
        if (ratio > MAX_DEPTH_RATIO && !depthsPerImage[i].empty()) {
            auto &depths = depthsPerImage[i];
            // depths 已排序，取中位数
            double median = depths[depths.size() / 2];
            // 用 median/3 和 median*3 作为范围（9 倍跨度，足够覆盖场景深度变化）
            r.first = std::max(0.01, median / 3.0);
            r.second = median * 3.0;
            ++clampedCount;
        }
    }

    // 全局兜底
    double globalMin = std::numeric_limits<double>::max();
    double globalMax = 0.0;
    for (const auto &r : ranges) {
        if (r.first > 0) {
            globalMin = std::min(globalMin, r.first);
            globalMax = std::max(globalMax, r.second);
        }
    }
    if (globalMin > globalMax) {
        globalMin = 0.1;
        globalMax = 100.0;
    }

    for (auto &r : ranges) {
        if (r.first < 0) r = {globalMin, globalMax};
    }

    return ranges;
}

// ═══════════════════════════════════════════════════════════════
// PatchMatch 核心算法
// ═══════════════════════════════════════════════════════════════

inline float PatchMatchReconstructorV2::sampleGrayBilinear(const cv::Mat &gray, float x, float y)
{
    // 越界 1 像素外返回 -1 表示无效（避免边界外推导致错误匹配）
    if (x < 0 || y < 0 || x > gray.cols - 1 || y > gray.rows - 1) return -1.0f;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, gray.cols - 1);
    int y1 = std::min(y0 + 1, gray.rows - 1);
    float dx = x - x0;
    float dy = y - y0;

    // 行指针直接索引，避免 4 次 Mat::at 的重复 step 计算
    const uint8_t *row0 = gray.ptr<uint8_t>(y0);
    const uint8_t *row1 = gray.ptr<uint8_t>(y1);

    float v00 = row0[x0];
    float v01 = row0[x1];
    float v10 = row1[x0];
    float v11 = row1[x1];

    return (1.0f - dx) * (1.0f - dy) * v00 +
           dx * (1.0f - dy) * v01 +
           (1.0f - dx) * dy * v10 +
           dx * dy * v11;
}

float PatchMatchReconstructorV2::computePixelCost(int x, int y, float depth,
                                                 float nx, float ny, float nz,
                                                 const Context &ctx) const
{
    // 对每个源视图计算 NCC 代价并聚合：
    //   costTopK == 1：赢者通吃（取 min，历史行为）
    //   costTopK  > 1：最优 k 个源视图的代价平均。被遮挡/高光/运动物体
    //                  污染的视图代价偏高、被挤出前 k，不参与聚合，
    //                  因此对遮挡边界和弱纹理更稳健（ACMM/COLMAP 思路）。
    float bestCost = 2.0f; // NCC 代价范围 [0, 2]

    const int topK = m_options.costTopK;
    // topK>1 时收集各源视图代价；thread_local 避免每次调用分配
    static thread_local std::vector<float> srcCosts;
    if (topK > 1) srcCosts.clear();

    const int r = m_options.windowRadius;
    const double sigmaSpatial = m_options.sigmaSpatial > 0 ? m_options.sigmaSpatial : double(r);
    const double sigmaColor = m_options.sigmaColor;
    const double invSs2 = 1.0 / (2.0 * sigmaSpatial * sigmaSpatial);
    const double invSc2 = 1.0 / (2.0 * sigmaColor * sigmaColor);

    const double fx = ctx.fx;
    const double fy = ctx.fy;
    const double cx = ctx.cx;
    const double cy = ctx.cy;

    // 计算 slanted plane 的平面距离 planeD
    // 平面方程: n · P = planeD，其中 P 是参考相机坐标系下的 3D 点
    // 对于像素 (x, y, depth): P = depth * K_ref^-1 * [x, y, 1]^T
    // planeD = depth * (nx*(x-cx)/fx + ny*(y-cy)/fy + nz)
    double nKinv = nx * (x - cx) / fx + ny * (y - cy) / fy + nz;
    if (std::abs(nKinv) < 1e-9) return 2.0f;
    double planeD = double(depth) * nKinv;
    if (std::abs(planeD) < 1e-9) return 2.0f;
    double invPlaneD = 1.0 / planeD;

    // ── 预计算参考图窗口 ────────────────────────────────────────────────
    // 双边权重与参考图像素值是源视图无关的循环不变量，原实现在每个源视图
    // 上重复计算（N_src × 窗口面积 次 exp）。此处提到循环外，数值结果
    // 与优化前逐位一致。详见 PatchMatchReconstructorV2.h 中 sfm::pm 的说明。
    sfm::pmv2::WindowCache &cache = sfm::pmv2::windowCache();
    const int winCount = sfm::pmv2::prepareWindow(ctx.refGray, x, y, r, invSs2, invSc2, cache);

    for (const auto &src : ctx.sources) {
        // 构造 slanted plane Homography:
        //   H = K_src · (R_rel + T_rel · n^T / planeD) · K_ref^-1
        // M = R_rel + T_rel · n^T / planeD
        // M[i][j] = R_rel[i][j] + T_rel[i] · n[j] / planeD
        const double Tx = src.T[0];
        const double Ty = src.T[1];
        const double Tz = src.T[2];

        // T_rel · n^T / planeD 的 9 个元素
        double Tn_pD00 = Tx * nx * invPlaneD;
        double Tn_pD01 = Tx * ny * invPlaneD;
        double Tn_pD02 = Tx * nz * invPlaneD;
        double Tn_pD10 = Ty * nx * invPlaneD;
        double Tn_pD11 = Ty * ny * invPlaneD;
        double Tn_pD12 = Ty * nz * invPlaneD;
        double Tn_pD20 = Tz * nx * invPlaneD;
        double Tn_pD21 = Tz * ny * invPlaneD;
        double Tn_pD22 = Tz * nz * invPlaneD;

        // M = R_rel + T_rel · n^T / planeD
        double M00 = src.R[0] + Tn_pD00;
        double M01 = src.R[1] + Tn_pD01;
        double M02 = src.R[2] + Tn_pD02;
        double M10 = src.R[3] + Tn_pD10;
        double M11 = src.R[4] + Tn_pD11;
        double M12 = src.R[5] + Tn_pD12;
        double M20 = src.R[6] + Tn_pD20;
        double M21 = src.R[7] + Tn_pD21;
        double M22 = src.R[8] + Tn_pD22;

        // KM = K_src · M
        const double k00 = src.K[0], k01 = src.K[1], k02 = src.K[2];
        const double k10 = src.K[3], k11 = src.K[4], k12 = src.K[5];
        const double k20 = src.K[6], k21 = src.K[7], k22 = src.K[8];

        double KM00 = k00 * M00 + k01 * M10 + k02 * M20;
        double KM01 = k00 * M01 + k01 * M11 + k02 * M21;
        double KM02 = k00 * M02 + k01 * M12 + k02 * M22;
        double KM10 = k10 * M00 + k11 * M10 + k12 * M20;
        double KM11 = k10 * M01 + k11 * M11 + k12 * M21;
        double KM12 = k10 * M02 + k11 * M12 + k12 * M22;
        double KM20 = k20 * M00 + k21 * M10 + k22 * M20;
        double KM21 = k20 * M01 + k21 * M11 + k22 * M21;
        double KM22 = k20 * M02 + k21 * M12 + k22 * M22;

        // H = KM · K_ref^-1
        // K_ref^-1 = [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]
        double H00 = KM00 / fx;
        double H01 = KM01 / fy;
        double H02 = -KM00 * cx / fx - KM01 * cy / fy + KM02;
        double H10 = KM10 / fx;
        double H11 = KM11 / fy;
        double H12 = -KM10 * cx / fx - KM11 * cy / fy + KM12;
        double H20 = KM20 / fx;
        double H21 = KM21 / fy;
        double H22 = -KM20 * cx / fx - KM21 * cy / fy + KM22;

        // 双边加权 NCC（窗口内所有像素用同一个 H warp，因为它们在同一个平面上）
        // 参考图像素值与双边权重已在循环外预计算到 cache，这里只做 warp + 采样 + 累加
        double sumRef = 0, sumSrc = 0, sumRefSq = 0, sumSrcSq = 0, sumRefSrc = 0;
        double sumW = 0;
        int validCount = 0;

        for (int k = 0; k < winCount; ++k) {
            const int rx = cache.rx[k];
            const int ry = cache.ry[k];

            // warp 到源图像
            double denom = H20 * rx + H21 * ry + H22;
            if (std::abs(denom) < 1e-9) continue;
            double invDenom = 1.0 / denom;
            double srcX = (H00 * rx + H01 * ry + H02) * invDenom;
            double srcY = (H10 * rx + H11 * ry + H12) * invDenom;

            float srcVal = sampleGrayBilinear(src.gray, float(srcX), float(srcY));
            if (srcVal < 0) continue;
            ++validCount;

            const float refVal = cache.refVal[k];
            const double w = cache.weight[k];

            sumRef += refVal * w;
            sumSrc += srcVal * w;
            sumRefSq += refVal * refVal * w;
            sumSrcSq += srcVal * srcVal * w;
            sumRefSrc += refVal * srcVal * w;
            sumW += w;
        }

        // 有效像素不足，跳过此源视图
        if (validCount < (2 * r + 1) * (2 * r + 1) / 4 || sumW < 1e-6) continue;

        // NCC = covar / sqrt(varRef * varSrc)
        // 注意：使用 E[x²]-E[x]² 公式在弱纹理区域数值不稳定（两个相近大数相减）
        // 参考 COLMAP patch_match_cuda.cu 的处理：
        //   1. kMinVar 早退：var < 1e-5 时直接返回 kMaxCost=2.0，避免除以极小值导致 NCC 爆炸
        //   2. 最终 clamp(1-NCC, 0, 2) 双重保护，防止数值误差产生负 cost
        double meanRef = sumRef / sumW;
        double meanSrc = sumSrc / sumW;
        double varRef = sumRefSq / sumW - meanRef * meanRef;
        double varSrc = sumSrcSq / sumW - meanSrc * meanSrc;
        double covar = sumRefSrc / sumW - meanRef * meanSrc;

        // COLMAP kMinVar = 1e-5f 是针对 float [0,1] 图像的
        // 我们使用 uint8 [0,255] 图像，方差范围大 255^2 倍，等效阈值为 1e-5 * 255^2 ≈ 0.65
        // 取 1.0 作为保守值，避免弱纹理区域数值不稳定
        constexpr double kMinVar = 1.0;
        if (varRef < kMinVar || varSrc < kMinVar) continue;

        double ncc = covar / std::sqrt(varRef * varSrc);
        // clamp NCC 到 [-1, 1]（数学上 NCC 范围，防止数值误差）
        ncc = std::max(-1.0, std::min(1.0, ncc));
        double cost = 1.0 - ncc;
        // COLMAP 风格双重 clamp，确保 cost ∈ [0, 2]
        cost = std::max(0.0, std::min(2.0, cost));
        if (topK > 1) {
            srcCosts.push_back(float(cost));
        } else if (cost < bestCost) {
            bestCost = float(cost);
        }
    }

    // top-k 聚合：最优 k 个视图代价的平均；可用视图不足 k 时退化为全部平均
    if (topK > 1) {
        return sfm::pmv2::aggregateTopK(srcCosts, topK);
    }

    return bestCost;
}

void PatchMatchReconstructorV2::processPixel(int x, int y,
                                           cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                                           const Context &ctx,
                                           bool propagateDown,
                                           const sfm::pmv2::PlanarPrior *prior,
                                           std::mt19937 &rng) const
{
    const int cols = ctx.cols;
    const int rows = ctx.rows;
    const double depthMin = ctx.depthMin;
    const double depthMax = ctx.depthMax;

    const double fx = ctx.fx;
    const double fy = ctx.fy;
    const double cx = ctx.cx;
    const double cy = ctx.cy;

    auto genRandom01 = [&rng]() -> float {
        return static_cast<float>(rng()) / static_cast<float>(rng.max());
    };
    auto genRandomSigned = [&genRandom01]() -> float {
        return 2.0f * genRandom01() - 1.0f;
    };

    const float depthRange = float(depthMax - depthMin);
    const float initialDispUpdate = depthRange * 0.5f;
    const float initialNormUpdate = 1.0f;
    const float refineStopThres = 0.1f;

    float currDepth = depthMap.at<float>(y, x);
    cv::Vec3f currNormal = normalMap.at<cv::Vec3f>(y, x);

    float bestDepth = currDepth;
    cv::Vec3f bestNormal = currNormal;
    float bestCost = computePixelCost(x, y, bestDepth,
                                      bestNormal[0], bestNormal[1], bestNormal[2],
                                      ctx);

    // ── 平面先验（ACMP 受限后验）─────────────────────────────────
    // 有先验的像素不再取 min cost，而是最大化
    //   restricted = exp(-cost²/β) · (γ + exp(-Δd²/2σd²) · exp(-Δa²/2σa²))
    // 其中 Δd/Δa 是候选假设与先验平面的深度差/法向夹角。
    // 弱纹理区各假设光度代价接近，先验项成为决定性因素 → 深度贴向
    // 邻域三角面片拟合出的平面，完整度提升；纹理区光度项主导，不受影响。
    // 参数与 ACMP 一致：γ=0.5, β=0.18, σd=(dmax-dmin)/64, σa=5°。
    const bool hasPrior = prior && prior->mask.at<uint8_t>(y, x) != 0;
    float priorD = 0.0f;
    cv::Vec3f priorN(0.0f, 0.0f, 1.0f);
    const float priorGamma = 0.5f;
    const float priorBeta = 0.18f;
    const float priorSigmaD = depthRange / 64.0f;
    const float priorSigmaA = 3.14159265f * (5.0f / 180.0f);
    float invTwoSigmaD2 = 0.0f, invTwoSigmaA2 = 0.0f;
    float bestScore = 0.0f;   // 受限后验得分（仅 hasPrior 时使用）

    if (hasPrior) {
        priorD = prior->depth.at<float>(y, x);
        priorN = prior->normal.at<cv::Vec3f>(y, x);
        invTwoSigmaD2 = 1.0f / (2.0f * priorSigmaD * priorSigmaD);
        invTwoSigmaA2 = 1.0f / (2.0f * priorSigmaA * priorSigmaA);
    }

    auto restrictedScore = [&](float cost, float d, const cv::Vec3f &n) -> float {
        const float dd = d - priorD;
        float dot = priorN[0] * n[0] + priorN[1] * n[1] + priorN[2] * n[2];
        dot = std::max(-1.0f, std::min(1.0f, dot));
        const float da = std::acos(dot);
        const float priorTerm = priorGamma
                              + std::exp(-dd * dd * invTwoSigmaD2)
                              * std::exp(-da * da * invTwoSigmaA2);
        return std::exp(-cost * cost / priorBeta) * priorTerm;
    };

    // 统一的候选接受逻辑：无先验 = min cost（与历史逐位一致），
    // 有先验 = max restricted（confidence 仍记录原始代价，供过滤用）
    auto accept = [&](float cost, float d, const cv::Vec3f &n) {
        if (!hasPrior) {
            if (cost < bestCost) {
                bestCost = cost;
                bestDepth = d;
                bestNormal = n;
            }
            return;
        }
        const float s = restrictedScore(cost, d, n);
        if (s > bestScore) {
            bestScore = s;
            bestCost = cost;
            bestDepth = d;
            bestNormal = n;
        }
    };

    if (hasPrior) {
        bestScore = restrictedScore(bestCost, bestDepth, bestNormal);
    }

    // 传播：根据 propagateDown 选择邻居方向
    // propagateDown=true: 读左 (x-1,y) + 上 (x,y-1) → 从左上向右下传播
    // propagateDown=false: 读右 (x+1,y) + 下 (x,y+1) → 从右下向左上传播
    struct NbSpec { int nx, ny; };
    NbSpec neighbors[2] = {
        {x + (propagateDown ? -1 : 1), y},
        {x, y + (propagateDown ? -1 : 1)}
    };

    for (const auto &nb : neighbors) {
        if (nb.nx < 0 || nb.nx >= cols || nb.ny < 0 || nb.ny >= rows) continue;

        float nbDepth = depthMap.at<float>(nb.ny, nb.nx);
        cv::Vec3f nbNormal = normalMap.at<cv::Vec3f>(nb.ny, nb.nx);
        if (nbDepth <= 0) continue;

        double nKinv_nb = nbNormal[0] * (nb.nx - cx) / fx +
                          nbNormal[1] * (nb.ny - cy) / fy +
                          nbNormal[2];
        if (std::abs(nKinv_nb) < 1e-9) continue;
        double planeD = double(nbDepth) * nKinv_nb;

        double nKinv_curr = nbNormal[0] * (x - cx) / fx +
                            nbNormal[1] * (y - cy) / fy +
                            nbNormal[2];
        if (std::abs(nKinv_curr) < 1e-9) continue;
        float newDepth = float(planeD / nKinv_curr);

        if (newDepth < depthMin || newDepth > depthMax) continue;

        float cost = computePixelCost(x, y, newDepth,
                                      nbNormal[0], nbNormal[1], nbNormal[2],
                                      ctx);
        accept(cost, newDepth, nbNormal);
    }

    // 先验平面假设本身作为候选（ACMP PlaneHypothesisRefinement 的核心项）：
    // 让弱纹理像素即使传播不到好邻居，也能贴上先验平面
    if (hasPrior) {
        const float cost = computePixelCost(x, y, priorD,
                                            priorN[0], priorN[1], priorN[2], ctx);
        accept(cost, priorD, priorN);

        // 先验引导的随机候选：深度 ~ U(priorD ± 3σd)，法向在 σa 内扰动
        float d = priorD + 3.0f * priorSigmaD * genRandomSigned();
        d = std::max(float(depthMin), std::min(float(depthMax), d));
        cv::Vec3f n = priorN + cv::Vec3f(priorSigmaA * genRandomSigned(),
                                         priorSigmaA * genRandomSigned(),
                                         priorSigmaA * genRandomSigned());
        const float nlen = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (nlen > 1e-6f) {
            n /= nlen;
            if (n[2] <= 0.1f) { n[2] = 0.1f; n /= std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]); }
            const float cost2 = computePixelCost(x, y, d, n[0], n[1], n[2], ctx);
            accept(cost2, d, n);
        }
    }

    // 退火式随机细化：while 循环，每步 1 个样本，范围减半
    {
        float dispUpdate = initialDispUpdate;
        float normUpdate = initialNormUpdate;

        while (dispUpdate > refineStopThres) {
            float deltaD = dispUpdate * genRandomSigned();
            float newDepth = bestDepth + deltaD;
            if (newDepth < depthMin || newDepth > depthMax) {
                dispUpdate *= 0.5f;
                normUpdate *= 0.5f;
                continue;
            }

            cv::Vec3f deltaN(normUpdate * genRandomSigned(),
                             normUpdate * genRandomSigned(),
                             normUpdate * genRandomSigned());
            cv::Vec3f newNormal = bestNormal + deltaN;
            float nlen = std::sqrt(newNormal[0]*newNormal[0] +
                                   newNormal[1]*newNormal[1] +
                                   newNormal[2]*newNormal[2]);
            if (nlen < 1e-6f) {
                dispUpdate *= 0.5f;
                normUpdate *= 0.5f;
                continue;
            }
            newNormal /= nlen;

            if (newNormal[2] <= 0.1f) {
                dispUpdate *= 0.5f;
                normUpdate *= 0.5f;
                continue;
            }

            float cost = computePixelCost(x, y, newDepth,
                                          newNormal[0], newNormal[1], newNormal[2],
                                          ctx);
            accept(cost, newDepth, newNormal);

            dispUpdate *= 0.5f;
            normUpdate *= 0.5f;
        }
    }

    depthMap.at<float>(y, x) = bestDepth;
    normalMap.at<cv::Vec3f>(y, x) = bestNormal;
    confidence.at<float>(y, x) = bestCost;
}

cv::Mat PatchMatchReconstructorV2::runPatchMatch(const Context &ctx, cv::Mat &confidence,
                                               cv::Mat &normalMap,
                                               const cv::Mat &initDepth,
                                               const cv::Mat &initNormal,
                                               int numIters) const
{
    const int rows = ctx.rows;
    const int cols = ctx.cols;
    const double depthMin = ctx.depthMin;
    const double depthMax = ctx.depthMax;

    cv::Mat depthMap(rows, cols, CV_32FC1);
    normalMap.create(rows, cols, CV_32FC3);  // 法向图 (nx, ny, nz)
    confidence.create(rows, cols, CV_32FC1);

    const bool hasInit = !initDepth.empty() && !initNormal.empty()
                         && initDepth.size() == depthMap.size()
                         && initNormal.size() == normalMap.size();

    // 1. 初始化深度和法向
    // 有初始图时：有效像素直接采用，无效像素走随机初始化。
    // 无初始图时（单尺度或最粗层）：全图随机初始化。
    // 随机序列与历史版本逐位一致的前提是：无初始图时逐像素调用 rng 的
    // 次序完全不变——因此这里对"无初始图"路径保持原样，对"有初始图"
    // 路径只对无效像素调用 rng（此时本来也不存在可比的历史基线）。
    {
        std::mt19937 rng(42);
        auto genRandom01 = [&]() -> float {
            return static_cast<float>(rng()) / static_cast<float>(rng.max());
        };
        // 随机深度 [depthMin, depthMax]
        auto genRandomDepth = [&]() -> float {
            return float(depthMin) + float(depthMax - depthMin) * genRandom01();
        };
        // 随机单位法向，z > 0（指向相机）
        auto genRandomNormal = [&]() -> cv::Vec3f {
            float nx, ny, nz;
            do {
                nx = 2.0f * genRandom01() - 1.0f;
                ny = 2.0f * genRandom01() - 1.0f;
                nz = 2.0f * genRandom01() - 1.0f;
            } while (nx*nx + ny*ny + nz*nz < 0.01f || nz <= 0.1f);
            float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
            return cv::Vec3f(nx / nlen, ny / nlen, nz / nlen);
        };

        int reused = 0;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (hasInit) {
                    const float d = initDepth.at<float>(y, x);
                    if (d > 0.0f && std::isfinite(d)) {
                        depthMap.at<float>(y, x) = d;
                        normalMap.at<cv::Vec3f>(y, x) = initNormal.at<cv::Vec3f>(y, x);
                        ++reused;
                        continue;
                    }
                }
                depthMap.at<float>(y, x) = genRandomDepth();
                normalMap.at<cv::Vec3f>(y, x) = genRandomNormal();
            }
        }
        if (hasInit) {
            log(QString("[PatchMatch] 初始化: 复用上采样结果 %1/%2 像素, 其余随机")
                    .arg(reused).arg(rows * cols));
        }
    }

    log(QString("[PatchMatch] 初始化: %1x%2, 深度范围 [%3, %4], %5 源视图, %6 轮")
            .arg(cols).arg(rows)
            .arg(depthMin, 0, 'f', 2).arg(depthMax, 0, 'f', 2)
            .arg(ctx.sources.size()).arg(numIters));

    // 2. 红黑网格并行迭代传播 + 退火式随机细化
    // 将像素按 (x+y)%2 分为红(0)/黑(1)两组，红格的4邻域都是黑格，反之亦然。
    // 每轮迭代分两阶段：
    //   阶段1：并行处理先序组（读后序组的旧值，写先序组的新值）
    //   阶段2：并行处理后序组（读先序组的新值，写后序组的新值）
    // 偶数轮 propagateDown=true（读左+上），先红后黑
    // 奇数轮 propagateDown=false（读右+下），先黑后红
    // 这样保持了原算法的双向传播语义，且红黑分离天然无读写冲突。

    // 线程池由 reconstruct() 创建，通过 ctx.pool 传入，整个重建过程复用
    if (!ctx.pool) {
        log(QStringLiteral("[PatchMatch] 错误: 线程池未设置"));
        return depthMap;
    }
    sfm::ThreadPool &pool = *ctx.pool;

    const int numThreads = static_cast<int>(pool.thread_count());
    const int stripHeight = std::max(1, rows / (numThreads * 4));  // 每线程约4个strip，保证负载均衡

    QElapsedTimer iterTimer;
    iterTimer.start();

    // ── 平面先验（ACMP 两阶段思路的多尺度适配）─────────────────────
    // 每层先跑 warmup 轮纯光度迭代收敛出可信深度，再从当前深度图+代价图
    // 提取支撑点 → Delaunay → 平面光栅化，剩余轮次用受限后验继续迭代。
    // 与 ACMP 跑两遍完整 PatchMatch 相比不增加迭代预算。
    // planePrior=false 时 warmupIters=numIters、priorPtr 恒为 nullptr，
    // processPixel 走纯光度路径，与历史行为逐位一致。
    const bool wantPrior = m_options.planePrior && numIters >= 2;
    const int warmupIters = wantPrior ? std::max(1, numIters / 2) : numIters;
    sfm::pmv2::PlanarPrior planarPrior;
    const sfm::pmv2::PlanarPrior *priorPtr = nullptr;

    for (int iter = 0; iter < numIters; ++iter) {
        if (wantPrior && iter == warmupIters && !priorPtr) {
            QElapsedTimer priorTimer;
            priorTimer.start();
            planarPrior = sfm::pmv2::buildPlanarPrior(
                depthMap, confidence, ctx.fx, ctx.fy, ctx.cx, ctx.cy,
                m_options.planePriorStep, m_options.planePriorMaxCost,
                depthMin, depthMax);
            const int coverage = cv::countNonZero(planarPrior.mask);
            if (coverage > 0) {
                priorPtr = &planarPrior;
                log(QString("[PatchMatch] 平面先验: %1 支撑点, 覆盖 %2/%3 像素 (%4%), %5 ms")
                        .arg(planarPrior.supportCount)
                        .arg(coverage).arg(rows * cols)
                        .arg(100.0 * coverage / (rows * cols), 0, 'f', 1)
                        .arg(priorTimer.elapsed()));
            } else {
                log(QString("[PatchMatch] 平面先验: 支撑点不足(%1)，本层退回纯光度")
                        .arg(planarPrior.supportCount));
            }
        }

        const bool propagateDown = (iter % 2 == 0);
        // 先序组：propagateDown时先红(0)，否则先黑(1)
        const int firstParity = propagateDown ? 0 : 1;

        for (int phase = 0; phase < 2; ++phase) {
            const int parity = (firstParity + phase) % 2;

            // 按行分 strip 提交到线程池
            std::vector<std::future<void>> futures;
            for (int yStart = 0; yStart < rows; yStart += stripHeight) {
                int yEnd = std::min(yStart + stripHeight, rows);
                int stripIdx = yStart / stripHeight;

                futures.push_back(pool.submit(
                    [this, yStart, yEnd, cols, parity, propagateDown, stripIdx,
                     &depthMap, &normalMap, &confidence, &ctx, priorPtr]() {
                        std::mt19937 rng(12345u + static_cast<unsigned>(stripIdx));
                        for (int y = yStart; y < yEnd; ++y) {
                            for (int x = 0; x < cols; ++x) {
                                if (((x + y) & 1) != parity) continue;
                                processPixel(x, y, depthMap, normalMap, confidence,
                                             ctx, propagateDown, priorPtr, rng);
                            }
                        }
                    }));
            }
            // 等待所有 strip 完成（红黑阶段同步点）
            for (auto &f : futures) f.wait();
        }

        // 诊断统计：深度分布和平均代价
        {
            int validCnt = 0;
            double sumDepth = 0, sumCost = 0;
            float minD = std::numeric_limits<float>::max();
            float maxD = std::numeric_limits<float>::lowest();
            for (int yy = 0; yy < rows; ++yy) {
                for (int xx = 0; xx < cols; ++xx) {
                    float d = depthMap.at<float>(yy, xx);
                    float c = confidence.at<float>(yy, xx);
                    if (d > 0 && std::isfinite(d)) {
                        ++validCnt;
                        sumDepth += d;
                        minD = std::min(minD, d);
                        maxD = std::max(maxD, d);
                    }
                    sumCost += c;
                }
            }
            double meanDepth = validCnt > 0 ? sumDepth / validCnt : 0.0;
            double meanCost = sumCost / (rows * cols);
            log(QString("[PatchMatch] 迭代 %1/%2 完成 (%3 ms, 有效=%4%, 深度[%5,%6] 均值=%7, 均代价=%8)")
                    .arg(iter + 1).arg(numIters)
                    .arg(iterTimer.elapsed())
                    .arg(100.0 * validCnt / (rows * cols), 0, 'f', 1)
                    .arg(minD, 0, 'f', 2).arg(maxD, 0, 'f', 2)
                    .arg(meanDepth, 0, 'f', 2)
                    .arg(meanCost, 0, 'f', 4));
        }
        iterTimer.restart();
    }

    // 3. 过滤：代价过高的像素置零
    const float costThreshold = 1.0f - float(m_options.filterMinNcc);
    int filteredCount = 0;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (confidence.at<float>(y, x) > costThreshold) {
                depthMap.at<float>(y, x) = 0.0f;
                ++filteredCount;
            }
        }
    }
    log(QString("[PatchMatch] 代价过滤: 阈值=%1, 过滤 %2 像素 (%3%)")
            .arg(costThreshold, 0, 'f', 3).arg(filteredCount)
            .arg(100.0 * filteredCount / (rows * cols), 0, 'f', 1));

    return depthMap;
}

void PatchMatchReconstructorV2::jointBilateralUpsample(
    const cv::Mat &depthLow, const cv::Mat &normalLow,
    const cv::Mat &grayHigh,
    cv::Mat &depthHigh, cv::Mat &normalHigh) const
{
    const int colsH = grayHigh.cols;
    const int rowsH = grayHigh.rows;
    const int colsL = depthLow.cols;
    const int rowsL = depthLow.rows;

    // 尺寸比（pyrDown 下非严格 2 倍，如 749 -> 375）
    const double scaleX = static_cast<double>(colsH) / colsL;
    const double scaleY = static_cast<double>(rowsH) / rowsL;

    depthHigh.create(rowsH, colsH, CV_32FC1);
    normalHigh.create(rowsH, colsH, CV_32FC3);

    // σs 以低分辨率像素为单位，3×3 邻域下 1.0 覆盖足够；
    // σr 以灰度 [0,255] 为单位，20 大致对应 ACMM 在 [0,1] 浮点图上的 0.08。
    const double inv2s2 = 1.0 / (2.0 * 1.0 * 1.0);
    const double inv2r2 = 1.0 / (2.0 * 20.0 * 20.0);

    for (int y = 0; y < rowsH; ++y) {
        const uint8_t *gRow = grayHigh.ptr<uint8_t>(y);
        float *dRow = depthHigh.ptr<float>(y);
        cv::Vec3f *nRow = normalHigh.ptr<cv::Vec3f>(y);

        for (int x = 0; x < colsH; ++x) {
            // 高分辨率像素 p 对应的低分辨率连续坐标（像素中心约定）
            const double plx = (x + 0.5) / scaleX - 0.5;
            const double ply = (y + 0.5) / scaleY - 0.5;
            const int qx0 = static_cast<int>(std::lround(plx));
            const int qy0 = static_cast<int>(std::lround(ply));

            const double Iref = static_cast<double>(gRow[x]);

            double sumW = 0.0, sumD = 0.0;
            double sumNx = 0.0, sumNy = 0.0, sumNz = 0.0;

            for (int dy = -1; dy <= 1; ++dy) {
                const int qy = qy0 + dy;
                if (qy < 0 || qy >= rowsL) continue;
                const float *dLRow = depthLow.ptr<float>(qy);
                const cv::Vec3f *nLRow = normalLow.ptr<cv::Vec3f>(qy);

                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = qx0 + dx;
                    if (qx < 0 || qx >= colsL) continue;

                    const float dL = dLRow[qx];
                    if (!(dL > 0.0f) || !std::isfinite(dL)) continue;

                    // 颜色核采样高分辨率引导图（边缘保持的关键）：
                    // 低分辨率像素 q 中心对应到高分辨率的位置，最近邻取灰度
                    const double hx = (qx + 0.5) * scaleX - 0.5;
                    const double hy = (qy + 0.5) * scaleY - 0.5;
                    const int hxI = std::min(std::max(static_cast<int>(hx + 0.5), 0), colsH - 1);
                    const int hyI = std::min(std::max(static_cast<int>(hy + 0.5), 0), rowsH - 1);
                    const double Iq = static_cast<double>(grayHigh.at<uint8_t>(hyI, hxI));

                    const double dsx = plx - qx;
                    const double dsy = ply - qy;
                    const double dc = Iref - Iq;
                    const double w = std::exp(-(dsx * dsx + dsy * dsy) * inv2s2
                                              - dc * dc * inv2r2);

                    sumW += w;
                    sumD += w * dL;
                    const cv::Vec3f &nL = nLRow[qx];
                    sumNx += w * nL[0];
                    sumNy += w * nL[1];
                    sumNz += w * nL[2];
                }
            }

            if (sumW > 1e-9) {
                dRow[x] = static_cast<float>(sumD / sumW);
                cv::Vec3f n(static_cast<float>(sumNx),
                            static_cast<float>(sumNy),
                            static_cast<float>(sumNz));
                const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                nRow[x] = (len > 1e-6f) ? (n / len) : cv::Vec3f(0.0f, 0.0f, 1.0f);
            } else {
                // 邻域全无效：置 0，由后续传播的邻居或随机初始化填补
                dRow[x] = 0.0f;
                nRow[x] = cv::Vec3f(0.0f, 0.0f, 1.0f);
            }
        }
    }
}

namespace {

/// 视图的刚性几何参数，展平为 POD 供内层循环直接索引（避免 cv::Mat 间接寻址）
struct ViewGeom {
    double K[9] = {};   ///< 内参 K[r*3+c]
    double R[9] = {};   ///< world→camera 旋转 R[r*3+c]
    double Rt[9] = {};  ///< Rᵀ，即 camera→world 旋转
    double t[3] = {};   ///< world→camera 平移
    double C[3] = {};   ///< 相机中心（世界坐标）= −Rᵀ·t

    bool valid = false;
};

/// Xc = R·Xw + t（9 乘加，不经过任何 cv::Mat）
inline void worldToCamera(const ViewGeom &g, const double Xw[3], double Xc[3])
{
    Xc[0] = g.R[0] * Xw[0] + g.R[1] * Xw[1] + g.R[2] * Xw[2] + g.t[0];
    Xc[1] = g.R[3] * Xw[0] + g.R[4] * Xw[1] + g.R[5] * Xw[2] + g.t[1];
    Xc[2] = g.R[6] * Xw[0] + g.R[7] * Xw[1] + g.R[8] * Xw[2] + g.t[2];
}

/// Xw = Rᵀ·Xc + C（等价于 R⁻¹·(Xc − t)，但省掉一次减法与一次矩阵乘）
inline void cameraToWorld(const ViewGeom &g, const double Xc[3], double Xw[3])
{
    Xw[0] = g.Rt[0] * Xc[0] + g.Rt[1] * Xc[1] + g.Rt[2] * Xc[2] + g.C[0];
    Xw[1] = g.Rt[3] * Xc[0] + g.Rt[4] * Xc[1] + g.Rt[5] * Xc[2] + g.C[1];
    Xw[2] = g.Rt[6] * Xc[0] + g.Rt[7] * Xc[1] + g.Rt[8] * Xc[2] + g.C[2];
}

}  // namespace

void PatchMatchReconstructorV2::filterGeometricConsistency(
    const DenseReconstructionInput &input,
    std::vector<StereoPairResult> &results,
    sfm::ThreadPool &pool) const
{
    const int numRes = static_cast<int>(results.size());
    if (numRes < 2) {
        log(QStringLiteral("[PatchMatch] 视图数不足，跳过几何一致性过滤"));
        return;
    }

    // ── 1. 收集各视图几何 ──────────────────────────────────────────────
    std::vector<ViewGeom> geoms(numRes);
    for (int i = 0; i < numRes; ++i) {
        const int imgIdx = results[i].imageId1;
        if (imgIdx < 0 || imgIdx >= static_cast<int>(input.cameraPoses.size())) continue;

        const auto &pose = input.cameraPoses[imgIdx];
        const cv::Mat &Kmat = pose.K.empty() ? input.cameraMatrix : pose.K;
        if (Kmat.empty() || pose.R.empty() || pose.t.empty()) continue;

        ViewGeom &g = geoms[i];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                g.K[r * 3 + c] = Kmat.at<double>(r, c);
                g.R[r * 3 + c] = pose.R.at<double>(r, c);
                g.Rt[r * 3 + c] = pose.R.at<double>(c, r);  // Rᵀ
            }
            g.t[r] = pose.t.at<double>(r);
        }
        const cv::Mat C = -pose.R.t() * pose.t;
        for (int r = 0; r < 3; ++r) g.C[r] = C.at<double>(r);
        g.valid = true;
    }

    // ── 2. 为每个视图挑选邻居：相机中心最近的若干张 ───────────────────
    // 航测（俯视、近似共面）下，相机中心距离是稳定且与待估深度无关的近邻度量，
    // 不需要额外的启发式。
    constexpr int kMaxNeighbors = 8;
    std::vector<std::vector<int>> neighbors(numRes);
    for (int i = 0; i < numRes; ++i) {
        if (!geoms[i].valid) continue;
        std::vector<std::pair<double, int>> cand;
        for (int j = 0; j < numRes; ++j) {
            if (i == j || !geoms[j].valid) continue;
            if (results[j].depth.empty()) continue;
            const double dx = geoms[j].C[0] - geoms[i].C[0];
            const double dy = geoms[j].C[1] - geoms[i].C[1];
            const double dz = geoms[j].C[2] - geoms[i].C[2];
            cand.emplace_back(dx * dx + dy * dy + dz * dz, j);
        }
        std::sort(cand.begin(), cand.end());
        for (const auto &[dist, j] : cand) {
            neighbors[i].push_back(j);
            if (static_cast<int>(neighbors[i].size()) >= kMaxNeighbors) break;
        }
    }

    const double maxReprojPx = m_options.filterGeomConsistencyMaxCost;
    const int minConsistent = std::max(1, m_options.filterMinNumConsistent);

    // ── 3. 逐视图统计一致源视图数（并行于行，结果写 mask）─────────────
    std::vector<cv::Mat> keepMask(numRes);
    for (int i = 0; i < numRes; ++i) {
        if (results[i].depth.empty()) continue;
        keepMask[i] = cv::Mat::zeros(results[i].depth.size(), CV_8UC1);
    }

    long long totalValid = 0;
    long long totalKept = 0;

    for (int i = 0; i < numRes; ++i) {
        if (!geoms[i].valid || results[i].depth.empty()) continue;
        if (static_cast<int>(neighbors[i].size()) < minConsistent) {
            // 邻居不够，无法判定，整图保留（不惩罚，只记录）
            keepMask[i].setTo(1);
            continue;
        }

        const cv::Mat &Dref = results[i].depth;
        const ViewGeom &gi = geoms[i];
        const double fx = gi.K[0], fy = gi.K[4], cx = gi.K[2], cy = gi.K[5];

        std::vector<std::future<void>> futures;
        std::vector<std::atomic<long long>> validCnt(
            Dref.rows), keptCnt(Dref.rows);  // per-row 统计，避免锁竞争
        for (int r = 0; r < Dref.rows; ++r) { validCnt[r] = 0; keptCnt[r] = 0; }

        const int stripHeight = std::max(1, Dref.rows / static_cast<int>(pool.thread_count() * 4));
        const auto &nbrs = neighbors[i];

        for (int yStart = 0; yStart < Dref.rows; yStart += stripHeight) {
            const int yEnd = std::min(yStart + stripHeight, Dref.rows);
            futures.push_back(pool.submit(
                [&, yStart, yEnd]() {
                    for (int y = yStart; y < yEnd; ++y) {
                        const float *drow = Dref.ptr<float>(y);
                        uint8_t *mrow = keepMask[i].ptr<uint8_t>(y);
                        long long rowValid = 0, rowKept = 0;

                        for (int x = 0; x < Dref.cols; ++x) {
                            const float dRef = drow[x];
                            if (!(dRef > 0.0f) || !std::isfinite(dRef)) continue;
                            ++rowValid;

                            // 参考光线上一点：Xw
                            double XcRef[3] = { (x - cx) * dRef / fx,
                                               (y - cy) * dRef / fy,
                                               double(dRef) };
                            double Xw[3];
                            cameraToWorld(gi, XcRef, Xw);

                            int numConsistent = 0;
                            for (int j : nbrs) {
                                const cv::Mat &Dsrc = results[j].depth;
                                const ViewGeom &gj = geoms[j];

                                // 投影到源视图
                                double XcSrc[3];
                                worldToCamera(gj, Xw, XcSrc);
                                if (XcSrc[2] <= 1e-6) continue;
                                const double uj = gj.K[0] * XcSrc[0] / XcSrc[2] + gj.K[2];
                                const double vj = gj.K[4] * XcSrc[1] / XcSrc[2] + gj.K[5];
                                const int ujI = static_cast<int>(uj + 0.5);
                                const int vjI = static_cast<int>(vj + 0.5);
                                if (ujI < 0 || ujI >= Dsrc.cols || vjI < 0 || vjI >= Dsrc.rows) continue;

                                const float dSrc = Dsrc.ptr<float>(vjI)[ujI];
                                if (!(dSrc > 0.0f) || !std::isfinite(dSrc)) continue;

                                // 源视图自身深度反投影 → 世界 → 投回参考视图
                                double XcSrc2[3] = { (ujI - gj.K[2]) * dSrc / gj.K[0],
                                                    (vjI - gj.K[5]) * dSrc / gj.K[4],
                                                    double(dSrc) };
                                double Xw2[3];
                                cameraToWorld(gj, XcSrc2, Xw2);
                                double XcBack[3];
                                worldToCamera(gi, Xw2, XcBack);
                                if (XcBack[2] <= 1e-6) continue;
                                const double ub = gi.K[0] * XcBack[0] / XcBack[2] + gi.K[2];
                                const double vb = gi.K[4] * XcBack[1] / XcBack[2] + gi.K[5];

                                const double errPx = std::hypot(ub - x, vb - y);
                                if (errPx < maxReprojPx) ++numConsistent;
                            }

                            if (numConsistent >= minConsistent) {
                                mrow[x] = 1;
                                ++rowKept;
                            }
                        }

                        validCnt[y] = rowValid;
                        keptCnt[y] = rowKept;
                    }
                }));
        }
        for (auto &f : futures) f.wait();

        long long v = 0, k = 0;
        for (int r = 0; r < Dref.rows; ++r) { v += validCnt[r].load(); k += keptCnt[r].load(); }
        totalValid += v;
        totalKept += k;
    }

    // ── 4. 应用 mask ───────────────────────────────────────────────────
    for (int i = 0; i < numRes; ++i) {
        if (results[i].depth.empty() || keepMask[i].empty()) continue;
        float *d = results[i].depth.ptr<float>();
        const uint8_t *m = keepMask[i].ptr<uint8_t>();
        const size_t n = static_cast<size_t>(results[i].depth.rows) * results[i].depth.cols;
        for (size_t p = 0; p < n; ++p) {
            if (!m[p]) d[p] = 0.0f;
        }
    }

    log(QString("[PatchMatch] 几何一致性过滤: 阈值=%1 px, 最少一致视图=%2, 保留 %3 / %4 (%5%)")
            .arg(maxReprojPx, 0, 'f', 2).arg(minConsistent)
            .arg(totalKept).arg(totalValid)
            .arg(totalValid > 0 ? 100.0 * totalKept / totalValid : 0.0, 0, 'f', 1));
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

    const int refIdx = problem.refImageIdx;
    if (refIdx < 0 || refIdx >= input.imageFiles.size() ||
        refIdx >= static_cast<int>(input.cameraPoses.size())) {
        return result;
    }

    // 加载参考图像
    cv::Mat refImage = loadImage(input.imageFiles[refIdx]);
    if (refImage.empty()) return result;

    const auto &refPose = input.cameraPoses[refIdx];
    cv::Mat K_ref = refPose.K.empty() ? input.cameraMatrix.clone() : refPose.K.clone();

    // 缩放图像（maxImageSize <= 0 表示不限制，按原始分辨率处理；
    // 多尺度开启时全分辨率成本可控——粗层收敛后细层只需少量迭代）
    double scale = 1.0;
    int maxDim = std::max(refImage.cols, refImage.rows);
    if (m_options.maxImageSize > 0 && maxDim > m_options.maxImageSize) {
        scale = static_cast<double>(m_options.maxImageSize) / maxDim;
    }
    cv::Mat refImageScaled;
    cv::resize(refImage, refImageScaled, cv::Size(), scale, scale, cv::INTER_AREA);

    // 调整内参
    cv::Mat K_scaled = K_ref.clone();
    K_scaled.at<double>(0, 0) *= scale;
    K_scaled.at<double>(0, 2) *= scale;
    K_scaled.at<double>(1, 1) *= scale;
    K_scaled.at<double>(1, 2) *= scale;

    // 转换为灰度
    cv::Mat refGray;
    cv::cvtColor(refImageScaled, refGray, cv::COLOR_BGR2GRAY);

    // 准备源视图
    Context ctx;
    ctx.refGray = refGray;
    ctx.K_ref = K_scaled;
    ctx.fx = K_scaled.at<double>(0, 0);
    ctx.fy = K_scaled.at<double>(1, 1);
    ctx.cx = K_scaled.at<double>(0, 2);
    ctx.cy = K_scaled.at<double>(1, 2);
    ctx.rows = refGray.rows;
    ctx.cols = refGray.cols;
    ctx.refImageIdx = refIdx;
    ctx.pool = &pool;  // 复用 reconstruct() 创建的线程池

    // 深度范围：缩放后的像素对应的世界深度不变（深度是世界坐标系的Z）
    ctx.depthMin = depthRange.first > 0 ? depthRange.first : 0.1;
    ctx.depthMax = depthRange.second > 0 ? depthRange.second : 100.0;

    // 构建源视图数据
    const cv::Mat R_ref_inv = refPose.R.t();  // 与源视图无关，提到循环外
    for (int srcIdx : problem.srcImageIdxs) {
        if (srcIdx < 0 || srcIdx >= input.imageFiles.size() ||
            srcIdx >= static_cast<int>(input.cameraPoses.size())) continue;

        SourceView src;

        // 加载源图像并缩放
        cv::Mat srcImage = loadImage(input.imageFiles[srcIdx]);
        if (srcImage.empty()) continue;
        cv::Mat srcImageScaled;
        cv::resize(srcImage, srcImageScaled, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::cvtColor(srcImageScaled, src.gray, cv::COLOR_BGR2GRAY);

        // 源相机内参（缩放）
        const auto &srcPose = input.cameraPoses[srcIdx];
        cv::Mat K_src = srcPose.K.empty() ? input.cameraMatrix.clone() : srcPose.K.clone();
        K_src.at<double>(0, 0) *= scale;
        K_src.at<double>(0, 2) *= scale;
        K_src.at<double>(1, 1) *= scale;
        K_src.at<double>(1, 2) *= scale;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) src.K[i * 3 + j] = K_src.at<double>(i, j);
        }

        // 相对位姿: R_rel = R_src * R_ref^T, T_rel = t_src - R_src * R_ref^T * t_ref
        // 展平为 POD 数组，供热点路径 computePixelCost 直接索引
        cv::Mat R_rel = srcPose.R * R_ref_inv;
        cv::Mat T_rel = srcPose.t - srcPose.R * R_ref_inv * refPose.t;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) src.R[i * 3 + j] = R_rel.at<double>(i, j);
            src.T[i] = T_rel.at<double>(i);
        }
        src.imageIdx = srcIdx;

        ctx.sources.push_back(std::move(src));
    }

    if (ctx.sources.empty()) {
        log("[PatchMatch] 无有效源视图，跳过");
        return result;
    }

    log(QString("[PatchMatch] 参考 image %1: 工作尺寸 %2x%3, %4 源视图, 深度 [%5, %6]")
            .arg(refIdx).arg(ctx.cols).arg(ctx.rows).arg(ctx.sources.size())
            .arg(ctx.depthMin, 0, 'f', 2).arg(ctx.depthMax, 0, 'f', 2));

    // ── 运行 PatchMatch（单尺度或多尺度 + JBU）─────────────────────────
    cv::Mat confidence;
    cv::Mat depth;
    cv::Mat normalMap;

    // 决定层数：最粗层短边不小于 128 像素，最多 maxScales 层
    int numScales = 1;
    if (m_options.multiscale) {
        int w = ctx.cols, h = ctx.rows;
        while (numScales < m_options.maxScales && std::min(w, h) / 2 >= 128) {
            w /= 2;
            h /= 2;
            ++numScales;
        }
    }

    if (numScales <= 1) {
        depth = runPatchMatch(ctx, confidence, normalMap,
                              cv::Mat(), cv::Mat(), m_options.numIterations);
    } else {
        // 建金字塔 contexts（索引 0 = 最细 … numScales-1 = 最粗）。
        // 逐层 pyrDown 灰度；内参按实际尺寸比缩放（像素中心约定）；
        // R_rel / T_rel 与分辨率无关，随 Context 拷贝共享。
        std::vector<Context> pyr(numScales);
        pyr[0] = ctx;
        for (int lv = 1; lv < numScales; ++lv) {
            const Context &prev = pyr[lv - 1];
            Context &c = pyr[lv];
            c = prev;  // 拷贝深度范围/pool/R/T 等；gray 随后替换为本层数据

            cv::pyrDown(prev.refGray, c.refGray);
            c.rows = c.refGray.rows;
            c.cols = c.refGray.cols;
            const double sx = static_cast<double>(c.cols) / prev.cols;
            const double sy = static_cast<double>(c.rows) / prev.rows;

            c.fx = prev.fx * sx;
            c.fy = prev.fy * sy;
            c.cx = (prev.cx + 0.5) * sx - 0.5;
            c.cy = (prev.cy + 0.5) * sy - 0.5;
            c.K_ref = (cv::Mat_<double>(3, 3) << c.fx, 0, c.cx,
                                                 0, c.fy, c.cy,
                                                 0, 0, 1);

            for (size_t s = 0; s < c.sources.size(); ++s) {
                cv::Mat g;
                cv::pyrDown(prev.sources[s].gray, g);
                c.sources[s].gray = g;
                // 源内参同样缩放（行主序 K[0]=fx, K[2]=cx, K[4]=fy, K[5]=cy）
                c.sources[s].K[0] = prev.sources[s].K[0] * sx;
                c.sources[s].K[2] = (prev.sources[s].K[2] + 0.5) * sx - 0.5;
                c.sources[s].K[4] = prev.sources[s].K[4] * sy;
                c.sources[s].K[5] = (prev.sources[s].K[5] + 0.5) * sy - 0.5;
            }
        }

        // 各层迭代轮数：最粗层 numIterations，最细层 refineIterations，中间线性插值
        auto itersOfLevel = [&](int lv) -> int {
            if (lv == numScales - 1) return m_options.numIterations;
            if (lv == 0) return m_options.refineIterations;
            return static_cast<int>(std::lround(
                m_options.refineIterations
                + double(m_options.numIterations - m_options.refineIterations)
                      * lv / (numScales - 1)));
        };

        log(QString("[PatchMatch] 多尺度: %1 层 (最细 %2x%3 -> 最粗 %4x%5)")
                .arg(numScales).arg(pyr[0].cols).arg(pyr[0].rows)
                .arg(pyr[numScales - 1].cols).arg(pyr[numScales - 1].rows));

        for (int lv = numScales - 1; lv >= 0; --lv) {
            const int iters = itersOfLevel(lv);
            log(QString("[PatchMatch] ── 层级 %1/%2 (%3x%4), %5 轮 ──")
                    .arg(numScales - lv).arg(numScales)
                    .arg(pyr[lv].cols).arg(pyr[lv].rows).arg(iters));
            if (depth.empty()) {
                depth = runPatchMatch(pyr[lv], confidence, normalMap,
                                      cv::Mat(), cv::Mat(), iters);
            } else {
                cv::Mat upDepth, upNormal;
                jointBilateralUpsample(depth, normalMap, pyr[lv].refGray,
                                       upDepth, upNormal);
                depth = runPatchMatch(pyr[lv], confidence, normalMap,
                                      upDepth, upNormal, iters);
            }
        }
    }

    // 反缩放到原始尺寸
    // 注意：depth 与 confidence 必须一起缩放。之前只缩放了 depth，导致
    // confidence 停留在工作分辨率而上采样后的 depth 是全分辨率，两者行列数对不上。
    // 目前下游还没用到 confidence 所以没暴露，但一旦有人读写就是越界访问。
    // 两者都是逐像素标量，用 INTER_NEAREST 避免插值出物理上不存在的深度/代价。
    if (scale < 1.0) {
        cv::resize(depth, depth, refImage.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(confidence, confidence, refImage.size(), 0, 0, cv::INTER_NEAREST);
    }

    // 统计有效像素
    int validCount = 0;
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float d = depth.at<float>(y, x);
            if (d > 0 && std::isfinite(d)) ++validCount;
        }
    }
    log(QString("[PatchMatch] 深度图统计: %1x%2, 有效像素 %3 (%4%)")
            .arg(depth.cols).arg(depth.rows).arg(validCount)
            .arg(100.0 * validCount / (depth.rows * depth.cols), 0, 'f', 1));

    result.depth = depth;
    result.confidence = confidence;
    result.rectifiedImage1 = refImage;
    result.K_rect = K_ref.clone();
    result.R1_rect = cv::Mat::eye(3, 3, CV_64FC1);
    return result;
}

cv::Mat PatchMatchReconstructorV2::fuseSourceDepthMaps(
    const cv::Mat &refImage,
    const std::vector<cv::Mat> &srcDepths,
    const std::vector<cv::Mat> &srcConfidences) const
{
    (void)refImage;
    (void)srcDepths;
    (void)srcConfidences;
    // PatchMatch 已经是多视图联合优化，无需再融合
    // 保留方法以满足接口，但不会在 reconstruct 中调用
    return cv::Mat();
}

void PatchMatchReconstructorV2::fuseDepthMaps(
    const DenseReconstructionInput &input,
    const std::vector<StereoPairResult> &results,
    sfm::ThreadPool &pool,
    pcl::PointCloud<pcl::PointXYZRGB> &cloud) const
{
    const int numRes = static_cast<int>(results.size());
    if (numRes == 0) return;

    auto t0 = std::chrono::steady_clock::now();

    // ── 1. 收集各视图几何（复用 ViewGeom）────────────────────────────
    std::vector<ViewGeom> geoms(numRes);
    std::vector<cv::Mat> images(numRes);
    for (int i = 0; i < numRes; ++i) {
        const int imgIdx = results[i].imageId1;
        if (imgIdx < 0 || imgIdx >= static_cast<int>(input.cameraPoses.size())) continue;

        const auto &pose = input.cameraPoses[imgIdx];
        const cv::Mat &Kmat = pose.K.empty() ? input.cameraMatrix : pose.K;
        if (Kmat.empty() || pose.R.empty() || pose.t.empty()) continue;
        if (results[i].depth.empty()) continue;

        ViewGeom &g = geoms[i];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                g.K[r * 3 + c] = Kmat.at<double>(r, c);
                g.R[r * 3 + c] = pose.R.at<double>(r, c);
                g.Rt[r * 3 + c] = pose.R.at<double>(c, r);  // Rᵀ
            }
            g.t[r] = pose.t.at<double>(r);
        }
        const cv::Mat C = -pose.R.t() * pose.t;
        for (int r = 0; r < 3; ++r) g.C[r] = C.at<double>(r);
        g.valid = true;

        images[i] = loadImage(input.imageFiles[imgIdx]);
    }

    // ── 2. 每个视图的近邻（相机中心最近 K 个，与几何一致性过滤同源）──
    constexpr int kMaxNeighbors = 8;
    std::vector<std::vector<int>> neighbors(numRes);
    for (int i = 0; i < numRes; ++i) {
        if (!geoms[i].valid) continue;
        std::vector<std::pair<double, int>> cand;
        for (int j = 0; j < numRes; ++j) {
            if (i == j || !geoms[j].valid) continue;
            const double dx = geoms[j].C[0] - geoms[i].C[0];
            const double dy = geoms[j].C[1] - geoms[i].C[1];
            const double dz = geoms[j].C[2] - geoms[i].C[2];
            cand.emplace_back(dx * dx + dy * dy + dz * dz, j);
        }
        std::sort(cand.begin(), cand.end());
        for (const auto &[dist, j] : cand) {
            neighbors[i].push_back(j);
            if (static_cast<int>(neighbors[i].size()) >= kMaxNeighbors) break;
        }
    }

    // ── 3. visited 标记：被先前视图合并掉的像素不再独立出点 ──────────
    // 视图间串行处理（外层循环），视图 j 的 visited 只会被 i<j 的视图
    // 置位、在处理视图 j 时只读，无并发冲突；视图内行条带并行，多个
    // 线程对同一 visited[j] 像素只写 1（benign race，最坏重复出点）。
    std::vector<cv::Mat> visited(numRes);
    for (int i = 0; i < numRes; ++i) {
        if (geoms[i].valid)
            visited[i] = cv::Mat::zeros(results[i].depth.rows, results[i].depth.cols, CV_8UC1);
    }

    const double relErrMax = m_options.fusionMaxRelDepthErr;
    const int minSupport = std::max(1, m_options.fusionMinSupport);

    size_t numEmitted = 0, numConsumed = 0, numLowSupport = 0;

    // ── 4. 逐视图融合（外层串行，行条带并行）─────────────────────────
    for (int i = 0; i < numRes; ++i) {
        if (!geoms[i].valid) continue;
        const cv::Mat &Di = results[i].depth;
        const cv::Mat &Ci = results[i].confidence;
        const cv::Mat &Ii = images[i];
        const ViewGeom &gi = geoms[i];
        const int rows = Di.rows, cols = Di.cols;
        const auto &nbrs = neighbors[i];

        size_t stripEmitted = 0, stripConsumed = 0, stripLowSup = 0;
        std::mutex m;
        std::vector<std::future<void>> futures;
        const int stripHeight = std::max(1, rows / static_cast<int>(pool.thread_count() * 4));

        for (int yStart = 0; yStart < rows; yStart += stripHeight) {
            const int yEnd = std::min(yStart + stripHeight, rows);
            futures.push_back(pool.submit([&, yStart, yEnd]() {
                size_t emitted = 0, consumed = 0, lowSup = 0;
                std::vector<pcl::PointXYZRGB> stripPts;
                stripPts.reserve((yEnd - yStart) * cols / 4);

                for (int y = yStart; y < yEnd; ++y) {
                    const float *dRow = Di.ptr<float>(y);
                    const float *cRow = Ci.empty() ? nullptr : Ci.ptr<float>(y);
                    const uint8_t *vRow = visited[i].ptr<uint8_t>(y);
                    for (int x = 0; x < cols; ++x) {
                        const float d = dRow[x];
                        if (!(d > 0.0f) || !std::isfinite(d)) continue;
                        if (vRow[x]) { ++consumed; continue; }

                        // 自身观测：反投影到世界
                        const double xc = (x - gi.K[2]) * d / gi.K[0];
                        const double yc = (y - gi.K[5]) * d / gi.K[4];
                        double px = gi.Rt[0] * xc + gi.Rt[1] * yc + gi.Rt[2] * d + gi.C[0];
                        double py = gi.Rt[3] * xc + gi.Rt[4] * yc + gi.Rt[5] * d + gi.C[1];
                        double pz = gi.Rt[6] * xc + gi.Rt[7] * yc + gi.Rt[8] * d + gi.C[2];

                        double w = 1.0 / (1.0 + (cRow ? double(cRow[x]) : 0.0));
                        double sumW = w, sumX = w * px, sumY = w * py, sumZ = w * pz;
                        double sumR = 0, sumG = 0, sumB = 0;
                        if (!Ii.empty() && Ii.type() == CV_8UC3) {
                            const cv::Vec3b c0 = Ii.at<cv::Vec3b>(y, x);
                            sumB = w * c0[0]; sumG = w * c0[1]; sumR = w * c0[2];
                        }
                        int support = 1;

                        // 近邻视图中的重复观测
                        for (int j : nbrs) {
                            const ViewGeom &gj = geoms[j];
                            const cv::Mat &Dj = results[j].depth;
                            // 世界 → 相机 j
                            const double Xj = gj.R[0] * px + gj.R[1] * py + gj.R[2] * pz + gj.t[0];
                            const double Yj = gj.R[3] * px + gj.R[4] * py + gj.R[5] * pz + gj.t[1];
                            const double Zj = gj.R[6] * px + gj.R[7] * py + gj.R[8] * pz + gj.t[2];
                            if (Zj <= 0.0) continue;
                            const double u = gj.K[0] * Xj / Zj + gj.K[2];
                            const double v = gj.K[4] * Yj / Zj + gj.K[5];
                            const int uj = static_cast<int>(u + 0.5);
                            const int vj = static_cast<int>(v + 0.5);
                            if (uj < 0 || uj >= Dj.cols || vj < 0 || vj >= Dj.rows) continue;
                            // 最近邻取深度（双线性会在遮挡边界混入前景/背景均值，反而不安全）
                            const float dj = Dj.at<float>(vj, uj);
                            if (!(dj > 0.0f) || !std::isfinite(dj)) continue;
                            if (std::abs(double(dj) - Zj) / double(dj) > relErrMax) continue;

                            // 同一表面：把视图 j 的观测也反投影进来做加权平均
                            const double xj = (u - gj.K[2]) * dj / gj.K[0];
                            const double yj = (v - gj.K[5]) * dj / gj.K[4];
                            const double qx = gj.Rt[0] * xj + gj.Rt[1] * yj + gj.Rt[2] * dj + gj.C[0];
                            const double qy = gj.Rt[3] * xj + gj.Rt[4] * yj + gj.Rt[5] * dj + gj.C[1];
                            const double qz = gj.Rt[6] * xj + gj.Rt[7] * yj + gj.Rt[8] * dj + gj.C[2];

                            const cv::Mat &Cfj = results[j].confidence;
                            const float cfj = Cfj.empty() ? 0.0f : Cfj.at<float>(vj, uj);
                            const double wj = 1.0 / (1.0 + double(cfj));
                            sumW += wj; sumX += wj * qx; sumY += wj * qy; sumZ += wj * qz;
                            const cv::Mat &Ij = images[j];
                            if (!Ij.empty() && Ij.type() == CV_8UC3) {
                                const cv::Vec3b cj = Ij.at<cv::Vec3b>(vj, uj);
                                sumB += wj * cj[0]; sumG += wj * cj[1]; sumR += wj * cj[2];
                            }
                            ++support;
                            visited[j].at<uint8_t>(vj, uj) = 1;  // 标记消费（benign race）
                        }

                        if (support < minSupport) { ++lowSup; continue; }

                        pcl::PointXYZRGB pt;
                        pt.x = float(sumX / sumW);
                        pt.y = float(sumY / sumW);
                        pt.z = float(sumZ / sumW);
                        pt.r = uint8_t(std::min(255.0, sumR / sumW + 0.5));
                        pt.g = uint8_t(std::min(255.0, sumG / sumW + 0.5));
                        pt.b = uint8_t(std::min(255.0, sumB / sumW + 0.5));
                        if (pt.r + pt.g + pt.b < 15) continue;  // 与旧路径一致的近黑点剔除
                        stripPts.push_back(pt);
                        ++emitted;
                    }
                }

                std::lock_guard<std::mutex> lk(m);
                cloud.points.insert(cloud.points.end(), stripPts.begin(), stripPts.end());
                stripEmitted += emitted; stripConsumed += consumed; stripLowSup += lowSup;
            }));
        }
        for (auto &f : futures) f.wait();
        numEmitted += stripEmitted; numConsumed += stripConsumed; numLowSupport += stripLowSup;
    }

    cloud.width = static_cast<uint32_t>(cloud.points.size());
    cloud.height = 1;
    cloud.is_dense = false;

    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    log(QString("[PatchMatch] 深度融合: %1 点 (被合并跳过 %2, 支持不足剔除 %3, relErr<%4, minSupport=%5, %6 s)")
            .arg(numEmitted).arg(numConsumed).arg(numLowSupport)
            .arg(relErrMax).arg(minSupport).arg(sec, 0, 'f', 1));
}

void PatchMatchReconstructorV2::backprojectDepth(
    const DenseReconstructionInput &input,
    int refIdx,
    const cv::Mat &depth,
    const cv::Mat &refImage,
    pcl::PointCloud<pcl::PointXYZRGB> &cloud) const
{
    if (refIdx < 0 || refIdx >= static_cast<int>(input.cameraPoses.size())) return;

    const auto &pose = input.cameraPoses[refIdx];
    const cv::Mat K = pose.K.empty() ? input.cameraMatrix.clone() : pose.K.clone();
    if (K.empty()) return;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    const cv::Mat R_world = pose.R.t();
    const cv::Mat t_world = -pose.R.t() * pose.t;

    constexpr int step = 2;

    for (int y = 0; y < depth.rows; y += step) {
        for (int x = 0; x < depth.cols; x += step) {
            float d = depth.at<float>(y, x);
            if (d <= 0.0f || !std::isfinite(d)) continue;

            double Xc = (x - cx) * d / fx;
            double Yc = (y - cy) * d / fy;
            double Zc = d;

            double Xw = R_world.at<double>(0, 0) * Xc + R_world.at<double>(0, 1) * Yc + R_world.at<double>(0, 2) * Zc + t_world.at<double>(0);
            double Yw = R_world.at<double>(1, 0) * Xc + R_world.at<double>(1, 1) * Yc + R_world.at<double>(1, 2) * Zc + t_world.at<double>(1);
            double Zw = R_world.at<double>(2, 0) * Xc + R_world.at<double>(2, 1) * Yc + R_world.at<double>(2, 2) * Zc + t_world.at<double>(2);

            uint8_t r = 128, g = 128, b = 128;
            if (!refImage.empty() && refImage.type() == CV_8UC3) {
                cv::Vec3b c = refImage.at<cv::Vec3b>(y, x);
                b = c[0]; g = c[1]; r = c[2];
            }

            if (r + g + b < 15) continue;

            pcl::PointXYZRGB pt;
            pt.x = static_cast<float>(Xw);
            pt.y = static_cast<float>(Yw);
            pt.z = static_cast<float>(Zw);
            pt.r = r; pt.g = g; pt.b = b;
            cloud.push_back(pt);
        }
    }
}
