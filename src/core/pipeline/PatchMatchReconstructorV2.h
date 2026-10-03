#ifndef PATCHMATCHRECONSTRUCTORV2_H
#define PATCHMATCHRECONSTRUCTORV2_H

#include "IDenseReconstructor.h"
#include "data/StereoTypes.h"

#include <QString>
#include <QMap>
#include <QVariant>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>   // cv::Subdiv2D（buildPlanarPrior 的 Delaunay 剖分）
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace sfm { class ThreadPool; }  // 前向声明，避免头文件暴露 ThreadPool 实现

namespace sfm {
namespace pmv2 {

/**
 * @brief 参考图窗口的每线程复用缓冲
 *
 * 匹配窗口内的参考图像素值与双边权重只依赖参考图自身和窗口偏移，
 * 与深度假设、法向、源视图都无关。原实现把它们的计算放在源视图循环内，
 * 导致每个源视图都重算一遍（N_src × 窗口面积 次 exp），而 exp 是代价函数
 * 里最昂贵的指令。提到循环外后数值结果完全等价，只是消除冗余计算。
 *
 * 用 thread_local 是因为代价函数被多线程调用，且缓冲区大小随 windowRadius
 * 变化，不适合放栈上固定数组。四个数组合并到同一个结构，保证每次调用
 * 只做一次 TLS 查找。
 */
struct WindowCache {
    std::vector<float> refVal;   ///< 窗口内参考图像素值
    std::vector<double> weight;  ///< 双边权重（保持 double，确保逐位一致）
    std::vector<int> rx;         ///< 窗口像素的参考图 x 坐标
    std::vector<int> ry;         ///< 窗口像素的参考图 y 坐标

    void ensure(int n)
    {
        if (static_cast<int>(refVal.size()) < n) {
            refVal.resize(n);
            weight.resize(n);
            rx.resize(n);
            ry.resize(n);
        }
    }
};

inline WindowCache &windowCache()
{
    static thread_local WindowCache cache;
    return cache;
}

/**
 * @brief 预计算参考图窗口内（图像界内）的像素值与双边权重
 *
 * @param refGray  参考灰度图（CV_8UC1）
 * @param x, y     窗口中心像素
 * @param r        窗口半径
 * @param invSs2   1/(2·sigmaSpatial²)
 * @param invSc2   1/(2·sigmaColor²)
 * @param cache    输出缓冲
 * @return 窗口内位于图像界内的像素个数
 */
inline int prepareWindow(const cv::Mat &refGray, int x, int y, int r,
                         double invSs2, double invSc2, WindowCache &cache)
{
    cache.ensure((2 * r + 1) * (2 * r + 1));

    const uint8_t refCenterVal = refGray.at<uint8_t>(y, x);
    int count = 0;

    for (int dy = -r; dy <= r; ++dy) {
        const int ry = y + dy;
        if (ry < 0 || ry >= refGray.rows) continue;
        const uint8_t *refRow = refGray.ptr<uint8_t>(ry);

        for (int dx = -r; dx <= r; ++dx) {
            const int rx = x + dx;
            if (rx < 0 || rx >= refGray.cols) continue;

            const float refVal = static_cast<float>(refRow[rx]);
            const double colorDiff = static_cast<double>(refVal)
                                   - static_cast<double>(refCenterVal);
            const double w = std::exp(-static_cast<double>(dx * dx + dy * dy) * invSs2
                                      - colorDiff * colorDiff * invSc2);

            cache.refVal[count] = refVal;
            cache.weight[count] = w;
            cache.rx[count] = rx;
            cache.ry[count] = ry;
            ++count;
        }
    }

    return count;
}

/**
 * @brief 多视图代价的 top-k 聚合：最优（最小）k 个代价的平均
 *
 * 语义：
 * - costs 为空 → 返回 maxCost（与 min 聚合的"无有效视图"行为一致）
 * - k >= costs.size() → 退化为全部代价的平均
 * - 否则取最小的 k 个代价求平均；k=1 时等价于 min（赢者通吃）
 *
 * 注意会就地重排 costs（nth_element），调用方不应依赖原顺序。
 */
inline float aggregateTopK(std::vector<float> &costs, int topK, float maxCost = 2.0f)
{
    if (costs.empty()) return maxCost;
    const int n = static_cast<int>(costs.size());
    const int k = std::max(1, std::min(topK, n));
    std::nth_element(costs.begin(), costs.begin() + (k - 1), costs.end());
    double sum = 0.0;
    for (int i = 0; i < k; ++i) sum += costs[i];
    return static_cast<float>(sum / k);
}

/**
 * @brief 平面先验（ACMP Planar Prior）：每像素的先验平面深度/法向
 *
 * depth(y,x) > 0 且 mask(y,x) != 0 表示该像素有先验平面。
 * normal 为单位向量、nz > 0（相机坐标系，指向相机，与 processPixel 约定一致）。
 */
struct PlanarPrior {
    cv::Mat depth;          ///< CV_32FC1，先验平面深度（0 = 无先验）
    cv::Mat normal;         ///< CV_32FC3，先验平面法向
    cv::Mat mask;           ///< CV_8UC1，先验有效掩码
    int supportCount = 0;   ///< 支撑点数量（诊断/日志用）
};

/**
 * @brief 从当前深度图 + 代价图构建平面先验
 *
 * ACMP 流程的 CPU 实现（参考 ACMP.cpp GetSupportPoints / DelaunayTriangulation /
 * GetPriorPlaneParams）：
 *   1. step×step 网格块内取代价最小且 < maxCost 的像素作为支撑点
 *   2. cv::Subdiv2D Delaunay 三角剖分
 *   3. 每个三角形反投影三顶点，叉积拟合平面（法向翻转到 nz>0）
 *   4. 重心坐标光栅化：平面深度 d = ρ / (n·K⁻¹·pix)，越出 [depthMin, depthMax] 作废
 *
 * 与 ACMP 的两处刻意差异：
 * - 平面拟合用叉积（3 点精确确定平面）代替 SVD solveZ，更便宜且确定性一致
 * - 法向约定 nz>0 指向相机（本工程约定），ACMP 是 w>0（等价于法向指离相机）
 *
 * 支撑点 < 3 或三角剖分失败时返回全空先验（mask 全 0），调用方按无先验处理。
 *
 * inline 放在头文件：patchmatch_test 不链接 sfmcore.lib（只编测试 main.cpp），
 * 与 prepareWindow/aggregateTopK 同一可测性约定。
 */
inline PlanarPrior buildPlanarPrior(const cv::Mat &depth, const cv::Mat &cost,
                                    double fx, double fy, double cx, double cy,
                                    int step, float maxCost,
                                    double depthMin, double depthMax)
{
    PlanarPrior prior;
    const int rows = depth.rows;
    const int cols = depth.cols;
    prior.depth = cv::Mat::zeros(rows, cols, CV_32FC1);
    prior.normal = cv::Mat::zeros(rows, cols, CV_32FC3);
    prior.mask = cv::Mat::zeros(rows, cols, CV_8UC1);

    // ── 1. 支撑点：step×step 块内取代价最小且 < maxCost 的有效深度像素 ──
    std::vector<cv::Point2f> support;
    step = std::max(1, step);
    for (int by = 0; by < rows; by += step) {
        const int ye = std::min(rows, by + step);
        for (int bx = 0; bx < cols; bx += step) {
            const int xe = std::min(cols, bx + step);
            float bestC = maxCost;
            cv::Point bestPt(-1, -1);
            for (int y = by; y < ye; ++y) {
                for (int x = bx; x < xe; ++x) {
                    const float d = depth.at<float>(y, x);
                    if (d <= 0.0f || !std::isfinite(d)) continue;
                    const float c = cost.at<float>(y, x);
                    if (c < bestC) {
                        bestC = c;
                        bestPt = cv::Point(x, y);
                    }
                }
            }
            if (bestPt.x >= 0) {
                support.emplace_back(static_cast<float>(bestPt.x),
                                     static_cast<float>(bestPt.y));
            }
        }
    }
    prior.supportCount = static_cast<int>(support.size());
    if (support.size() < 3) return prior;   // 全空先验

    // ── 2. Delaunay 三角剖分 ──
    std::vector<cv::Vec6f> tris;
    {
        cv::Subdiv2D subdiv(cv::Rect(0, 0, cols, rows));
        try {
            subdiv.insert(support);
            subdiv.getTriangleList(tris);
        } catch (const cv::Exception &) {
            return prior;   // 退化输入（共线等），按无先验处理
        }
    }

    const cv::Rect imageRC(0, 0, cols, rows);

    // ── 3. 逐三角形：反投影 → 叉积拟合平面 → 重心光栅化 ──
    auto backproject = [&](float px, float py, float d, double X[3]) {
        X[0] = (px - cx) * d / fx;
        X[1] = (py - cy) * d / fy;
        X[2] = d;
    };

    for (const auto &t : tris) {
        const cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
        // 与 ACMP 一致：三顶点都在图内的三角形才参与
        if (!imageRC.contains(p1) || !imageRC.contains(p2) || !imageRC.contains(p3)) continue;

        // 顶点坐标取整回像素（Subdiv2D 输出的就是支撑点坐标）
        const int x1 = cvRound(p1.x), y1 = cvRound(p1.y);
        const int x2 = cvRound(p2.x), y2 = cvRound(p2.y);
        const int x3 = cvRound(p3.x), y3 = cvRound(p3.y);
        const float d1 = depth.at<float>(y1, x1);
        const float d2 = depth.at<float>(y2, x2);
        const float d3 = depth.at<float>(y3, x3);
        if (d1 <= 0 || d2 <= 0 || d3 <= 0) continue;

        double X1[3], X2[3], X3[3];
        backproject(p1.x, p1.y, d1, X1);
        backproject(p2.x, p2.y, d2, X2);
        backproject(p3.x, p3.y, d3, X3);

        // 叉积拟合平面：n = (X2-X1)×(X3-X1)，翻转到 nz>0（指向相机）
        double e1[3] = {X2[0] - X1[0], X2[1] - X1[1], X2[2] - X1[2]};
        double e2[3] = {X3[0] - X1[0], X3[1] - X1[1], X3[2] - X1[2]};
        double nx = e1[1] * e2[2] - e1[2] * e2[1];
        double ny = e1[2] * e2[0] - e1[0] * e2[2];
        double nz = e1[0] * e2[1] - e1[1] * e2[0];
        const double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
        // 面积过小的退化三角形（近似共线）不可信
        if (nlen < 1e-9) continue;
        nx /= nlen; ny /= nlen; nz /= nlen;
        if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }

        // 平面方程 n·X = ρ，深度 d(x,y) = ρ / (nx·(x-cx)/fx + ny·(y-cy)/fy + nz)
        const double rho = nx * X1[0] + ny * X1[1] + nz * X1[2];

        // 重心坐标光栅化（包围盒裁剪到图内）
        const int minX = std::max(0, static_cast<int>(std::floor(std::min({p1.x, p2.x, p3.x}))));
        const int maxX = std::min(cols - 1, static_cast<int>(std::ceil(std::max({p1.x, p2.x, p3.x}))));
        const int minY = std::max(0, static_cast<int>(std::floor(std::min({p1.y, p2.y, p3.y}))));
        const int maxY = std::min(rows - 1, static_cast<int>(std::ceil(std::max({p1.y, p2.y, p3.y}))));

        // 边函数（2 倍有符号面积），退化时跳过
        const auto edge = [](const cv::Point2f &a, const cv::Point2f &b, float x, float y) {
            return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
        };
        const float area = edge(p1, p2, p3.x, p3.y);
        if (std::fabs(area) < 1e-6f) continue;

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                const float w1 = edge(p2, p3, static_cast<float>(x), static_cast<float>(y));
                const float w2 = edge(p3, p1, static_cast<float>(x), static_cast<float>(y));
                const float w3 = edge(p1, p2, static_cast<float>(x), static_cast<float>(y));
                // 同号（或面积为负时全负）即在三角形内；边上像素允许写入
                if (!((w1 >= 0 && w2 >= 0 && w3 >= 0) || (w1 <= 0 && w2 <= 0 && w3 <= 0))) continue;

                const double den = nx * (x - cx) / fx + ny * (y - cy) / fy + nz;
                if (std::fabs(den) < 1e-9) continue;
                const double d = rho / den;
                if (d < depthMin || d > depthMax) continue;

                prior.depth.at<float>(y, x) = static_cast<float>(d);
                prior.normal.at<cv::Vec3f>(y, x) = cv::Vec3f(
                    static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz));
                prior.mask.at<uint8_t>(y, x) = 1;
            }
        }
    }

    return prior;
}

} // namespace pmv2
} // namespace sfm

/**
 * @brief COLMAP PatchMatch 多视图密集重建器的优化版（CPU 实现，V2）
 *
 * 与第一版 PatchMatchReconstructor 并存（独立类、独立注册），在其基础上叠加：
 *   - 代价核热点优化：参考图窗口预计算 + 相机参数 POD 化
 *   - top-k 多视图代价聚合（costTopK，抗遮挡/高光）
 *   - 多尺度金字塔 + JBU 联合双边上采样（multiscale）
 *   - 多视图几何一致性过滤（geomConsistency，往返重投影剔除离群点）
 *   - 深度融合（fusion，跨视图去重 + 置信度加权平均）
 *   - 平面先验（planePrior，ACMP 受限后验，弱纹理区域更完整）
 *
 * 核心思路（参考 colmap::mvs::PatchMatch）：
 *   - 不做极线校正，直接在原始图像坐标系下用 Homography warping 匹配
 *   - 代价函数使用 NCC（对线性光照变化 a·I+b 不变），鲁棒于光照差异
 *   - 随机初始化 → 红黑棋盘邻域传播 → 随机细化（指数衰减扰动）
 */
class PatchMatchReconstructorV2 : public IDenseReconstructor
{
public:
    /**
     * @brief PatchMatch 参数集合
     *
     * 字段命名与默认值尽量与 COLMAP 的 PatchMatchOptions 保持一致，
     * 方便后续直接替换为 COLMAP CUDA 实现。
     */
    struct Options {
        double depthMin = -1.0;                  ///< 最小深度，<0 时从稀疏点自动估计
        double depthMax = -1.0;                  ///< 最大深度，<0 时从稀疏点自动估计
        double sigmaSpatial = -1.0;              ///< 空间权重sigma，<0 时取 windowRadius
        double sigmaColor = 51.0;                ///< 颜色权重sigma（NCC双边加权，uint8[0,255]图像）
        double nccSigma = 0.6;                   ///< NCC似然函数扩散系数
        double minTriangulationAngle = 0.5;      ///< 选源图像最小光轴夹角(度)，共轴数据集需调低
        double incidentAngleSigma = 0.9;         ///< 入射角似然函数扩散系数
        double geomConsistencyRegularizer = 0.3; ///< 几何一致性项相对权重（保留，暂未使用）
        double geomConsistencyMaxCost = 3.0;     ///< 几何一致性最大代价(像素)（保留，暂未使用）
        double filterMinNcc = 0.1;               ///< 过滤最小NCC阈值
        double filterMinTriangulationAngle = 3.0;///< 过滤最小三角化角度(度)
        double filterGeomConsistencyMaxCost = 1.0;///< 过滤几何一致性最大代价(像素)（保留，暂未使用）
        double cacheSize = 32.0;                 ///< 缓存大小(GB)（保留，CPU实现忽略）
        QString gpuIndex = "-1";                 ///< GPU索引，CPU实现忽略
        int maxImageSize = 1320;                 ///< 最大工作图像尺寸；<=0 不限制（全分辨率）
        int windowRadius = 5;                    ///< NCC窗口半径
        int windowStep = 1;                      ///< NCC窗口步长（保留，暂用1）
        int numSamples = 15;                     ///< 每像素随机采样数
        int numIterations = 5;                   ///< 传播迭代轮数（多尺度时为最粗层轮数）
        int costTopK = 4;                        ///< 多视图代价聚合：取最优 k 个源视图代价求平均
                                                 ///< 1 = 赢者通吃（min，历史行为）；
                                                 ///< k>1 时对遮挡/高光更稳健（被挡视图代价高，
                                                 ///< 不参与平均），代价整体水平会上升，需相应
                                                 ///< 放宽 filterGeomConsistencyMaxCost 等阈值。
                                                 ///< test14 全量验证默认 4（8 源取半）
        int filterMinNumConsistent = 2;          ///< 最小一致源视图数
        int maxScales = 3;                       ///< 多尺度最大层数（1 = 单尺度）
        int refineIterations = 2;                ///< 多尺度最细层迭代轮数（中间层线性插值）
        bool multiscale = true;                  ///< 是否启用多尺度 + JBU 上采样
        bool geomConsistency = true;             ///< 是否启用几何一致性过滤
        bool fusion = true;                      ///< 是否启用深度融合（跨视图去重+加权平均）
        double fusionMaxRelDepthErr = 0.02;      ///< 融合支持判定的最大相对深度误差
        int fusionMinSupport = 2;                ///< 融合最小支持观测数（含自身）
        bool planePrior = true;                  ///< 是否启用平面先验（ACMP 受限后验，每层迭代
                                                 ///< 前一半热身后构建先验、后一半使用）；
                                                 ///< test14 实测：一致率 +1.1pp、重投影 -9%，
                                                 ///< 弱纹理航测场景预期收益更大，耗时约 +12%
        float planePriorMaxCost = 0.2f;          ///< 支撑点最大代价（块内最小代价低于此值才入选）
        int planePriorStep = 5;                  ///< 支撑点采样网格步长（像素）
        bool filter = true;                      ///< 是否启用过滤
        bool allowMissingFiles = false;          ///< 是否允许缺失文件
        bool writeConsistencyGraph = false;      ///< 是否输出一致性图

        void loadFromParams(const QMap<QString, QVariant> &params);
    };

    /**
     * @brief PatchMatch 单视问题定义
     *
     * 参考 COLMAP PatchMatch::Problem：一个参考图像 + 若干源图像。
     */
    struct Problem {
        int refImageIdx = -1;           ///< 参考图像索引
        std::vector<int> srcImageIdxs;  ///< 源图像索引列表
    };

    PatchMatchReconstructorV2() = default;
    ~PatchMatchReconstructorV2() override = default;

    QString methodName() const override { return QStringLiteral("PatchMatchV2"); }
    QString displayName() const override { return QStringLiteral("PatchMatch 优化版（多尺度+平面先验）"); }

    void setConfig(const QMap<QString, QVariant> &params) override;
    DenseReconstructionOutput reconstruct(const DenseReconstructionInput &input) override;

private:
    /// 源视图预计算数据（避免重复加载和转换）
    ///
    /// 相机参数以 POD 数组（行主序）保存，而不是 cv::Mat：
    /// computePixelCost 是最内层热点（每像素 × 每迭代 × 每源视图），
    /// Mat::at<double>() 每次都要重算 data + step*i 间接寻址，
    /// 且 cv::Mat 头部会拖累 SourceView 的缓存局部性。
    struct SourceView {
        cv::Mat gray;       ///< 源灰度图（CV_8UC1）
        double K[9] = {};   ///< 源相机内参（已按缩放调整），K[r*3+c]
        double R[9] = {};   ///< 相对旋转 R_src * R_ref^T，R[r*3+c]
        double T[3] = {};   ///< 相对平移 t_src - R_src * R_ref^T * t_ref
        int imageIdx = -1;  ///< 源图像在 input.imageFiles 中的索引（诊断用）
    };

    /// PatchMatch 运行时上下文（避免长参数列表）
    struct Context {
        cv::Mat refGray;            ///< 参考灰度图
        cv::Mat K_ref;              ///< 参考内参（已缩放）
        double fx = 0.0, fy = 0.0;  ///< K_ref 的内参冗余缓存（热点路径避免 Mat::at）
        double cx = 0.0, cy = 0.0;
        std::vector<SourceView> sources;
        double depthMin = 0.0;
        double depthMax = 1.0;
        int rows = 0;
        int cols = 0;
        int refImageIdx = -1;       ///< 参考图像索引（诊断用）
        sfm::ThreadPool *pool = nullptr;  ///< 外部传入的线程池（由 reconstruct 创建，整个重建过程复用）
    };

    Options m_options;

    /// 为每个参考图像构建源视图集合
    std::vector<Problem> buildProblems(const DenseReconstructionInput &input) const;

    /// 估计每张图像的深度范围 [depthMin, depthMax]
    std::vector<std::pair<double, double>> estimateDepthRanges(
        const DenseReconstructionInput &input) const;

    /// 求解单个 PatchMatch 问题，输出参考视角的深度图结果
    StereoPairResult solveProblem(const DenseReconstructionInput &input,
                                  const Problem &problem,
                                  const std::pair<double, double> &depthRange,
                                  sfm::ThreadPool &pool) const;

    /// 运行 PatchMatch 核心算法：初始化 + 红黑传播 + 退火细化
    /// @param ctx 上下文（参考图、源视图、内参、深度范围）
    /// @param[out] confidence 每像素代价（越小越可信）
    /// @param[out] normalMap 每像素法向 (CV_32FC3)，多尺度层间传递用
    /// @param initDepth 可选初始深度（多尺度上采样结果）；为空或某像素 <=0 时
    ///                  对该像素走与单尺度完全一致的随机初始化（同种子同序列）
    /// @param initNormal 与 initDepth 配套的初始法向
    /// @param numIters 迭代轮数（多尺度时各层不同）
    /// @return 深度图 (CV_32FC1)
    cv::Mat runPatchMatch(const Context &ctx, cv::Mat &confidence, cv::Mat &normalMap,
                          const cv::Mat &initDepth, const cv::Mat &initNormal,
                          int numIters) const;

    /// 联合双边上采样（JBU）：低分辨率深度/法向 → 高分辨率
    ///
    /// 对高分辨率像素 p，在其对应的低分辨率 3×3 邻域内加权平均：
    ///   w = exp(-||pl - q||² / 2σs²) · exp(-(I_high(p) - I_high(q↑))² / 2σr²)
    /// 空间核在低分辨率坐标系度量，颜色核用**高分辨率**引导图（边缘保持的关键）。
    /// 低分辨率无效像素（深度 <= 0）不参与加权；邻域全无效时输出 0，
    /// 由后续传播的邻居传播或随机初始化填补。
    /// 法向用同样的权重加权后归一化（比简单双线性更贴边缘）。
    void jointBilateralUpsample(const cv::Mat &depthLow, const cv::Mat &normalLow,
                                const cv::Mat &grayHigh,
                                cv::Mat &depthHigh, cv::Mat &normalHigh) const;

    /// 多视图几何一致性过滤（需要在全部视角深度图算完后才能做）
    ///
    /// 对每个深度像素做正向-反向重投影：沿参考光线取点 X，投影到源视图得 (u,v)；
    /// 用源视图自身在该位置的深度反投影回世界坐标，再投影回参考视图，与原像素
    /// 比较（单位：像素）。一致源视图数不足 filterMinNumConsistent 的像素置零。
    ///
    /// 这一步能同时压制三类错误：遮挡边界的半遮挡像素、弱纹理区的多峰误匹配、
    /// 以及只在参考视图上成立的 floaters。代价是需要全部视角的深度图都在内存中，
    /// 且必须在光度代价之后再跑一遍 O(像素数 × 邻居数) 的重投影。
    /// @param results 各视角全分辨率深度图，原地过滤
    void filterGeometricConsistency(const DenseReconstructionInput &input,
                                    std::vector<StereoPairResult> &results,
                                    sfm::ThreadPool &pool) const;

    /// 处理单个像素：传播（读邻居）+ 退火式随机细化 + 写回
    /// 红黑网格并行化的基本单元，被 runPatchMatch 在红/黑两阶段中调用。
    /// @param propagateDown true=读左+上邻居, false=读右+下邻居（交替方向保证双向传播）
    /// @param prior 平面先验（nullptr 或该像素 mask=0 时走纯光度最小代价，
    ///              与无先验逐位一致）；有先验时改用受限后验 argmax 选择假设
    /// @param rng 每线程独立的随机数生成器
    void processPixel(int x, int y,
                      cv::Mat &depthMap, cv::Mat &normalMap, cv::Mat &confidence,
                      const Context &ctx,
                      bool propagateDown,
                      const sfm::pmv2::PlanarPrior *prior,
                      std::mt19937 &rng) const;

    /// 计算单像素在给定 (depth, normal) 平面假设下的多视图匹配代价
    /// @param depth 参考像素 (x, y) 的深度
    /// @param nx, ny, nz 法向（相机坐标系，单位向量，nz > 0 指向相机）
    /// @return [0, 2]，越小越一致
    float computePixelCost(int x, int y, float depth,
                           float nx, float ny, float nz,
                           const Context &ctx) const;

    /// 双线性采样灰度图（越界返回 -1 表示无效）
    static inline float sampleGrayBilinear(const cv::Mat &gray, float x, float y);

    /// 深度融合：跨视图去重 + 置信度加权平均（COLMAP fusion 思路）
    ///
    /// 非融合路径（backprojectDepth）把每个视图的深度图独立反投影、直接拼接，
    /// 同一表面在 N 个视图里就出现 N 次，且各副本位置略有差异（噪点增厚）。
    /// 本方法对每个有效深度像素：反投影 → 投影到 K 个近邻视图 → 深度一致的
    /// 邻居视为同一表面的重复观测，把所有观测（含自身）按 1/(1+代价) 加权
    /// 平均成**一个**点，并把被合并的邻居像素标记为已消费（不再独立出点）。
    /// 效果：点云去重变薄、位置噪声被多视图平均压低、颜色同步平均。
    /// @param results 各视角全分辨率深度图（须已完成几何一致性过滤）
    /// @param pool    行条带并行；视图间串行（visited 标记的消费顺序）
    void fuseDepthMaps(const DenseReconstructionInput &input,
                       const std::vector<StereoPairResult> &results,
                       sfm::ThreadPool &pool,
                       pcl::PointCloud<pcl::PointXYZRGB> &cloud) const;

    /// 将多张源视图匹配得到的深度图融合为一张
    cv::Mat fuseSourceDepthMaps(const cv::Mat &refImage,
                                const std::vector<cv::Mat> &srcDepths,
                                const std::vector<cv::Mat> &srcConfidences) const;

    /// 从参考图像深度图反投影到世界坐标系，追加到点云
    void backprojectDepth(const DenseReconstructionInput &input,
                          int refIdx,
                          const cv::Mat &depth,
                          const cv::Mat &refImage,
                          pcl::PointCloud<pcl::PointXYZRGB> &cloud) const;
};

#endif // PATCHMATCHRECONSTRUCTORV2_H
