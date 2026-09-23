#include "PatchMatchReconstructor.h"
#include "DenseReconstructionFactory.h"
#include "algorithm/ThreadPool.h"

#include <QElapsedTimer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
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
    numThreads = params.value("numThreads", numThreads).toInt();
    geomConsistency = params.value("geomConsistency", geomConsistency).toBool();
    filter = params.value("filter", filter).toBool();
    allowMissingFiles = params.value("allowMissingFiles", allowMissingFiles).toBool();
    writeConsistencyGraph = params.value("writeConsistencyGraph", writeConsistencyGraph).toBool();
}

void PatchMatchReconstructor::setConfig(const QMap<QString, QVariant> &params)
{
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

    log(QStringLiteral("[PatchMatch] 开始多视图密集重建 (Homography warping + NCC)"));

    // 创建线程池（整个密集重建过程复用，避免频繁创建/销毁）
    const int numThreads = m_options.numThreads > 0
        ? m_options.numThreads
        : static_cast<int>(std::thread::hardware_concurrency() / 2);
    sfm::ThreadPool pool(sfm::thread_pool_options{
        "patchmatch_pool",
        static_cast<std::size_t>(numThreads),
        true,
        0
    });
    log(QString("[PatchMatch] 并行: %1 线程, 红黑网格").arg(numThreads));

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

    // 4. 反投影生成密集点云
    for (const auto &result : results) {
        if (result.depth.empty()) continue;

        const int refIdx = result.imageId1;
        if (refIdx < 0 || refIdx >= input.imageFiles.size()) continue;

        cv::Mat refImage = loadImage(input.imageFiles[refIdx]);
        backprojectDepth(input, refIdx, result.depth, refImage, *output.denseCloud);
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

std::vector<std::pair<double, double>> PatchMatchReconstructor::estimateDepthRanges(
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

inline float PatchMatchReconstructor::sampleGrayBilinear(const cv::Mat &gray, float x, float y)
{
    // 越界 1 像素外返回 -1 表示无效（避免边界外推导致错误匹配）
    if (x < 0 || y < 0 || x > gray.cols - 1 || y > gray.rows - 1) return -1.0f;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, gray.cols - 1);
    int y1 = std::min(y0 + 1, gray.rows - 1);
    float dx = x - x0;
    float dy = y - y0;

    float v00 = gray.at<uint8_t>(y0, x0);
    float v01 = gray.at<uint8_t>(y0, x1);
    float v10 = gray.at<uint8_t>(y1, x0);
    float v11 = gray.at<uint8_t>(y1, x1);

    return (1.0f - dx) * (1.0f - dy) * v00 +
           dx * (1.0f - dy) * v01 +
           (1.0f - dx) * dy * v10 +
           dx * dy * v11;
}

float PatchMatchReconstructor::computePixelCost(int x, int y, float depth,
                                                 float nx, float ny, float nz,
                                                 const Context &ctx) const
{
    // 对每个源视图计算 NCC 代价，取最小值
    float bestCost = 2.0f; // NCC 代价范围 [0, 2]

    const int r = m_options.windowRadius;
    const double sigmaSpatial = m_options.sigmaSpatial > 0 ? m_options.sigmaSpatial : double(r);
    const double sigmaColor = m_options.sigmaColor;
    const double invSs2 = 1.0 / (2.0 * sigmaSpatial * sigmaSpatial);
    const double invSc2 = 1.0 / (2.0 * sigmaColor * sigmaColor);

    const cv::Mat &Kref = ctx.K_ref;
    const double fx = Kref.at<double>(0, 0);
    const double fy = Kref.at<double>(1, 1);
    const double cx = Kref.at<double>(0, 2);
    const double cy = Kref.at<double>(1, 2);

    // 计算 slanted plane 的平面距离 planeD
    // 平面方程: n · P = planeD，其中 P 是参考相机坐标系下的 3D 点
    // 对于像素 (x, y, depth): P = depth * K_ref^-1 * [x, y, 1]^T
    // planeD = depth * (nx*(x-cx)/fx + ny*(y-cy)/fy + nz)
    double nKinv = nx * (x - cx) / fx + ny * (y - cy) / fy + nz;
    if (std::abs(nKinv) < 1e-9) return 2.0f;
    double planeD = double(depth) * nKinv;
    if (std::abs(planeD) < 1e-9) return 2.0f;
    double invPlaneD = 1.0 / planeD;

    for (const auto &src : ctx.sources) {
        // 构造 slanted plane Homography:
        //   H = K_src · (R_rel + T_rel · n^T / planeD) · K_ref^-1
        // M = R_rel + T_rel · n^T / planeD
        // M[i][j] = R_rel[i][j] + T_rel[i] · n[j] / planeD
        const double Tx = src.T_rel.at<double>(0);
        const double Ty = src.T_rel.at<double>(1);
        const double Tz = src.T_rel.at<double>(2);

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
        double M00 = src.R_rel.at<double>(0, 0) + Tn_pD00;
        double M01 = src.R_rel.at<double>(0, 1) + Tn_pD01;
        double M02 = src.R_rel.at<double>(0, 2) + Tn_pD02;
        double M10 = src.R_rel.at<double>(1, 0) + Tn_pD10;
        double M11 = src.R_rel.at<double>(1, 1) + Tn_pD11;
        double M12 = src.R_rel.at<double>(1, 2) + Tn_pD12;
        double M20 = src.R_rel.at<double>(2, 0) + Tn_pD20;
        double M21 = src.R_rel.at<double>(2, 1) + Tn_pD21;
        double M22 = src.R_rel.at<double>(2, 2) + Tn_pD22;

        // KM = K_src · M
        const cv::Mat &Ks = src.K_src;
        const double k00 = Ks.at<double>(0, 0), k01 = Ks.at<double>(0, 1), k02 = Ks.at<double>(0, 2);
        const double k10 = Ks.at<double>(1, 0), k11 = Ks.at<double>(1, 1), k12 = Ks.at<double>(1, 2);
        const double k20 = Ks.at<double>(2, 0), k21 = Ks.at<double>(2, 1), k22 = Ks.at<double>(2, 2);

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
        double sumRef = 0, sumSrc = 0, sumRefSq = 0, sumSrcSq = 0, sumRefSrc = 0;
        double sumW = 0;
        int validCount = 0;

        uint8_t refCenterVal = ctx.refGray.at<uint8_t>(y, x);

        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int rx = x + dx;
                int ry = y + dy;
                if (rx < 0 || rx >= ctx.cols || ry < 0 || ry >= ctx.rows) continue;

                float refVal = ctx.refGray.at<uint8_t>(ry, rx);

                // warp 到源图像
                double denom = H20 * rx + H21 * ry + H22;
                if (std::abs(denom) < 1e-9) continue;
                double invDenom = 1.0 / denom;
                double srcX = (H00 * rx + H01 * ry + H02) * invDenom;
                double srcY = (H10 * rx + H11 * ry + H12) * invDenom;

                float srcVal = sampleGrayBilinear(src.gray, float(srcX), float(srcY));
                if (srcVal < 0) continue;
                ++validCount;

                // 双边权重：空间 + 颜色
                double spatialDist2 = double(dx * dx + dy * dy);
                double colorDiff = double(refVal) - double(refCenterVal);
                double colorDist2 = colorDiff * colorDiff;
                double w = std::exp(-spatialDist2 * invSs2 - colorDist2 * invSc2);

                sumRef += refVal * w;
                sumSrc += srcVal * w;
                sumRefSq += refVal * refVal * w;
                sumSrcSq += srcVal * srcVal * w;
                sumRefSrc += refVal * srcVal * w;
                sumW += w;
            }
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
        if (cost < bestCost) bestCost = float(cost);
    }

    return bestCost;
}

void PatchMatchReconstructor::processPixel(int x, int y,
                                           cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                                           const Context &ctx,
                                           bool propagateDown,
                                           std::mt19937 &rng) const
{
    const int cols = ctx.cols;
    const int rows = ctx.rows;
    const double depthMin = ctx.depthMin;
    const double depthMax = ctx.depthMax;

    const cv::Mat &Kref = ctx.K_ref;
    const double fx = Kref.at<double>(0, 0);
    const double fy = Kref.at<double>(1, 1);
    const double cx = Kref.at<double>(0, 2);
    const double cy = Kref.at<double>(1, 2);

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
        if (cost < bestCost) {
            bestCost = cost;
            bestDepth = newDepth;
            bestNormal = nbNormal;
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
            if (cost < bestCost) {
                bestCost = cost;
                bestDepth = newDepth;
                bestNormal = newNormal;
            }

            dispUpdate *= 0.5f;
            normUpdate *= 0.5f;
        }
    }

    depthMap.at<float>(y, x) = bestDepth;
    normalMap.at<cv::Vec3f>(y, x) = bestNormal;
    confidence.at<float>(y, x) = bestCost;
}

cv::Mat PatchMatchReconstructor::runPatchMatch(const Context &ctx, cv::Mat &confidence) const
{
    const int rows = ctx.rows;
    const int cols = ctx.cols;
    const double depthMin = ctx.depthMin;
    const double depthMax = ctx.depthMax;

    cv::Mat depthMap(rows, cols, CV_32FC1);
    cv::Mat normalMap(rows, cols, CV_32FC3);  // 法向图 (nx, ny, nz)
    confidence.create(rows, cols, CV_32FC1);

    // 内参（用于平面传播时计算像素深度）
    const cv::Mat &Kref = ctx.K_ref;
    const double fx = Kref.at<double>(0, 0);
    const double fy = Kref.at<double>(1, 1);
    const double cx = Kref.at<double>(0, 2);
    const double cy = Kref.at<double>(1, 2);

    // 1. 随机初始化深度和法向
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

        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                depthMap.at<float>(y, x) = genRandomDepth();
                normalMap.at<cv::Vec3f>(y, x) = genRandomNormal();
            }
        }
    }

    log(QString("[PatchMatch] 初始化: %1x%2, 深度范围 [%3, %4], %5 源视图")
            .arg(cols).arg(rows)
            .arg(depthMin, 0, 'f', 2).arg(depthMax, 0, 'f', 2)
            .arg(ctx.sources.size()));

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

    for (int iter = 0; iter < m_options.numIterations; ++iter) {
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
                     &depthMap, &normalMap, &confidence, &ctx]() {
                        std::mt19937 rng(12345u + static_cast<unsigned>(stripIdx));
                        for (int y = yStart; y < yEnd; ++y) {
                            for (int x = 0; x < cols; ++x) {
                                if (((x + y) & 1) != parity) continue;
                                processPixel(x, y, depthMap, normalMap, confidence,
                                             ctx, propagateDown, rng);
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
                    .arg(iter + 1).arg(m_options.numIterations)
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

StereoPairResult PatchMatchReconstructor::solveProblem(
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

    // 缩放图像（CPU 实现强制限制尺寸）
    double scale = 1.0;
    int maxDim = std::max(refImage.cols, refImage.rows);
    int targetSize = m_options.maxImageSize > 0 ? m_options.maxImageSize : 1000;
    if (maxDim > targetSize) {
        scale = static_cast<double>(targetSize) / maxDim;
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
    ctx.rows = refGray.rows;
    ctx.cols = refGray.cols;
    ctx.refImageIdx = refIdx;
    ctx.pool = &pool;  // 复用 reconstruct() 创建的线程池

    // 深度范围：缩放后的像素对应的世界深度不变（深度是世界坐标系的Z）
    ctx.depthMin = depthRange.first > 0 ? depthRange.first : 0.1;
    ctx.depthMax = depthRange.second > 0 ? depthRange.second : 100.0;

    // 构建源视图数据
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
        src.K_src = K_src;

        // 相对位姿: R_rel = R_src * R_ref^T, T_rel = t_src - R_src * R_ref^T * t_ref
        cv::Mat R_ref_inv = refPose.R.t();
        src.R_rel = srcPose.R * R_ref_inv;
        src.T_rel = srcPose.t - srcPose.R * R_ref_inv * refPose.t;
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

    // 运行 PatchMatch
    cv::Mat confidence;
    cv::Mat depth = runPatchMatch(ctx, confidence);

    // 反缩放到原始尺寸
    if (scale < 1.0) {
        cv::resize(depth, depth, refImage.size(), 0, 0, cv::INTER_NEAREST);
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

cv::Mat PatchMatchReconstructor::fuseSourceDepthMaps(
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

void PatchMatchReconstructor::backprojectDepth(
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
