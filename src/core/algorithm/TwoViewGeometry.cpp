#include "TwoViewGeometry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <opencv2/calib3d.hpp>

namespace
{

constexpr double kPi = 3.141592653589793;

/// 分阶段耗时探针（SFM_PAIR_TIMING=1 时输出到 stdout）
struct StageTimer
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double hMs = 0.0, eMs = 0.0, fMs = 0.0;
    void lapH() { hMs = elapsed(); restart(); }
    void lapE() { eMs = elapsed(); restart(); }
    void lapF() { fMs = elapsed(); restart(); }
    void dump(int n, int inliers) const
    {
        if (!std::getenv("SFM_PAIR_TIMING")) return;
        std::fprintf(stdout, "[几何计时] n=%zu H=%.1f E=%.1f F=%.1f inliers=%d\n",
                     n, hMs, eMs, fMs, inliers);
        std::fflush(stdout);
    }

private:
    double elapsed() const
    {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
    void restart() { t0 = std::chrono::steady_clock::now(); }
};

int countMask(const std::vector<uchar> &m)
{
    int c = 0;
    for (uchar v : m) {
        if (v) ++c;
    }
    return c;
}

/// 像素 → 归一化相机坐标（单位焦距，主点在原点）；只取前 count 个点
void normalizePoints(const std::vector<cv::Point2d> &in, double fx, double fy,
                     double cx, double cy, size_t count, std::vector<cv::Point2d> &out)
{
    out.clear();
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.emplace_back((in[i].x - cx) / fx, (in[i].y - cy) / fy);
    }
}

/// 等间隔抽样：内点率不变 → RANSAC 所需迭代次数不变，但每次迭代的
/// 打分成本降一个量级。抽样只用于「求模型」，内点统计仍在全量点上做。
void samplePairs(const std::vector<cv::Point2d> &p1, const std::vector<cv::Point2d> &p2,
                 size_t n, int cap, std::vector<cv::Point2d> &s1, std::vector<cv::Point2d> &s2)
{
    s1.clear();
    s2.clear();
    if (n == 0 || cap <= 0) {
        return;
    }
    const size_t stride = (n + static_cast<size_t>(cap) - 1) / static_cast<size_t>(cap);
    if (stride <= 1) {
        s1 = p1;
        s2 = p2;
        if (s1.size() > n) {
            s1.resize(n);
            s2.resize(n);
        }
        return;
    }
    s1.reserve(n / stride + 1);
    s2.reserve(n / stride + 1);
    for (size_t i = 0; i < n; i += stride) {
        s1.push_back(p1[i]);
        s2.push_back(p2[i]);
    }
}

/// 点到极线距离（像素）：l = [a b c]，d = |a·x + b·y + c| / √(a²+b²)
double lineDistance(const cv::Vec3d &l, const cv::Point2d &p)
{
    const double denom = std::sqrt(l[0] * l[0] + l[1] * l[1]);
    if (denom < 1e-12) {
        return std::numeric_limits<double>::max();
    }
    return std::fabs(l[0] * p.x + l[1] * p.y + l[2]) / denom;
}

/// 单应对称转移误差（像素）
double homographyError(const cv::Mat &H, const cv::Point2d &p1, const cv::Point2d &p2)
{
    if (H.empty()) {
        return std::numeric_limits<double>::max();
    }
    const double *h = H.ptr<double>();
    const double w = h[6] * p1.x + h[7] * p1.y + h[8];
    if (std::fabs(w) < 1e-12) {
        return std::numeric_limits<double>::max();
    }
    const double x = (h[0] * p1.x + h[1] * p1.y + h[2]) / w;
    const double y = (h[3] * p1.x + h[4] * p1.y + h[5]) / w;
    return std::hypot(x - p2.x, y - p2.y);
}

bool readK(const cv::Mat &K, double &fx, double &fy, double &cx, double &cy)
{
    if (K.empty() || K.rows != 3 || K.cols != 3) {
        return false;
    }
    if (K.type() != CV_64F) {
        return false;
    }
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
    return fx > 1e-6 && fy > 1e-6 && std::isfinite(cx) && std::isfinite(cy);
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════

double adaptiveRansacThresholdPx(int imageWidth, int imageHeight)
{
    const int maxDim = std::max(1, std::max(imageWidth, imageHeight));
    // 下限 2.0 px：阈值过紧（如 1000~2000px 图取 1~1.5px）会让内点率崩塌，
    // RANSAC 打满迭代上限还常配不出模型，慢且砍掉大量真实弱重叠对；
    // COLMAP 默认即固定 4 px。上限 4.0 与 COLMAP 对齐。
    return std::min(4.0, std::max(2.0, 0.001 * static_cast<double>(maxDim)));
}

TwoViewGeometry estimateTwoViewGeometry(const std::vector<cv::Point2d> &pts1,
                                        const std::vector<cv::Point2d> &pts2,
                                        const cv::Mat &K,
                                        const TwoViewGeometryOptions &opt)
{
    TwoViewGeometry g;
    const size_t n = std::min(pts1.size(), pts2.size());
    if (n < 8) {
        g.config = TWO_VIEW_DEGENERATE;
        return g;
    }

    const double thrPx = std::max(0.5, opt.ransacMaxErrorPx);
    const int minInliers = std::max(8, opt.minNumInliers);

    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    const bool hasK = !opt.forceUncalibrated && readK(K, fx, fy, cx, cy);

    StageTimer st;

    // ── 单应 H（始终估计：用于判定平面 / 纯旋转退化）────────────────────────
    // 抽样求模型 + 全量解析统计（对称转移误差 ≤ thrPx）
    std::vector<uchar> maskH(n, 0);
    int numH = 0;
    {
        std::vector<cv::Point2d> hs1, hs2;
        samplePairs(pts1, pts2, n, opt.ransacSampleCap, hs1, hs2);
        std::vector<uchar> maskHs;
        cv::Mat H = cv::findHomography(hs1, hs2, cv::RANSAC, thrPx, maskHs,
                                       opt.maxNumTrials, opt.confidence);
        if (!H.empty() && H.rows == 3 && H.cols == 3 && hs1.size() >= 4) {
            for (size_t i = 0; i < n; ++i) {
                if (homographyError(H, pts1[i], pts2[i]) <= thrPx) {
                    maskH[i] = 1;
                    ++numH;
                }
            }
            if (numH >= 8) {
                g.H = H;
            } else {
                numH = 0;
            }
        }
    }
    st.lapH();

    // ── 本质矩阵 E（已知内参时首选路径）────────────────────────────────────
    // 速度考虑：5 点法每次迭代都要对所有点打分，直接在两三千个匹配上跑 RANSAC
    // 会慢到几十分钟。做法是先按步长抽样（内点率不变 → 迭代次数不变），
    // 拿到 E 后再用解析式在【全量】点上统计内点，精度与召回都不打折。
    int numE = 0;
    std::vector<uchar> maskE, maskPose;
    cv::Mat E, R, t;
    if (hasK) {
        std::vector<cv::Point2d> s1, s2;
        samplePairs(pts1, pts2, n, opt.essentialRansacMaxPoints, s1, s2);
        // 5 点法要求归一化坐标（单位焦距）
        for (auto &p : s1) p = cv::Point2d((p.x - cx) / fx, (p.y - cy) / fy);
        for (auto &p : s2) p = cv::Point2d((p.x - cx) / fx, (p.y - cy) / fy);

        // 阈值换算到单位焦距空间：像素误差 / 平均焦距
        const double thrNorm = thrPx / (0.5 * (fx + fy));

        E = cv::findEssentialMat(s1, s2, 1.0, cv::Point2d(0.0, 0.0),
                                 cv::RANSAC, opt.confidence, thrNorm, maskE);
        if (!E.empty() && E.rows == 3 && E.cols == 3) {
            g.E = E;

            // 全量解析统计：归一化坐标下 x₂ₙ 到极线 l = E·x₁ₙ 的距离
            maskE.assign(n, 0);
            numE = 0;
            const double e00 = E.at<double>(0, 0), e01 = E.at<double>(0, 1), e02 = E.at<double>(0, 2);
            const double e10 = E.at<double>(1, 0), e11 = E.at<double>(1, 1), e12 = E.at<double>(1, 2);
            const double e20 = E.at<double>(2, 0), e21 = E.at<double>(2, 1), e22 = E.at<double>(2, 2);
            for (size_t i = 0; i < n; ++i) {
                const double x1n = (pts1[i].x - cx) / fx, y1n = (pts1[i].y - cy) / fy;
                const double x2n = (pts2[i].x - cx) / fx, y2n = (pts2[i].y - cy) / fy;
                const cv::Vec3d l(e00 * x1n + e01 * y1n + e02,
                                  e10 * x1n + e11 * y1n + e12,
                                  e20 * x1n + e21 * y1n + e22);
                if (lineDistance(l, cv::Point2d(x2n, y2n)) <= thrNorm) {
                    maskE[i] = 1;
                    ++numE;
                }
            }

            // 位姿只在内点上恢复：外点参与 cheirality 投票会把 R/t 带偏
            if (numE >= 8) {
                std::vector<cv::Point2d> i1, i2;
                i1.reserve(static_cast<size_t>(numE));
                i2.reserve(static_cast<size_t>(numE));
                for (size_t i = 0; i < n; ++i) {
                    if (!maskE[i]) continue;
                    i1.emplace_back((pts1[i].x - cx) / fx, (pts1[i].y - cy) / fy);
                    i2.emplace_back((pts2[i].x - cx) / fx, (pts2[i].y - cy) / fy);
                }
                try {
                    cv::recoverPose(E, i1, i2, R, t, 1.0, cv::Point2d(0.0, 0.0), maskPose);
                } catch (const cv::Exception &) {
                    R = cv::Mat();
                    t = cv::Mat();
                    maskPose.clear();
                }

                // recoverPose 的掩码是「内点子集」上的，映射回全量下标后供下游使用
                if (!maskPose.empty() && maskPose.size() == i1.size()) {
                    std::vector<uchar> full(n, 0);
                    size_t k = 0;
                    for (size_t i = 0; i < n; ++i) {
                        if (!maskE[i]) continue;
                        full[i] = maskPose[k++] ? 1 : 0;
                    }
                    maskPose = std::move(full);
                }
            }
        }
    }

    st.lapE();

    // ── 基础矩阵 F（仅当 E 路径不成立时才估，省一次 RANSAC）───────────────
    // 同样走「抽样求模型 + 全量解析统计」（对称极线距离）
    int numF = 0;
    std::vector<uchar> maskF;
    if (numE < minInliers) {
        std::vector<cv::Point2d> fs1, fs2;
        samplePairs(pts1, pts2, n, opt.ransacSampleCap, fs1, fs2);
        std::vector<uchar> maskFs;
        cv::Mat F = cv::findFundamentalMat(fs1, fs2, cv::FM_RANSAC, thrPx,
                                           opt.confidence, maskFs);
        if (!F.empty() && F.rows == 3 && F.cols == 3 && fs1.size() >= 8) {
            const double *f = F.ptr<double>();
            for (size_t i = 0; i < n; ++i) {
                const double x1 = pts1[i].x, y1 = pts1[i].y;
                const double x2 = pts2[i].x, y2 = pts2[i].y;
                // l₂ = F·x₁，l₁ = Fᵀ·x₂
                cv::Vec3d l2(f[0] * x1 + f[1] * y1 + f[2],
                             f[3] * x1 + f[4] * y1 + f[5],
                             f[6] * x1 + f[7] * y1 + f[8]);
                cv::Vec3d l1(f[0] * x2 + f[3] * y2 + f[6],
                             f[1] * x2 + f[4] * y2 + f[7],
                             f[2] * x2 + f[5] * y2 + f[8]);
                const double d2 = lineDistance(l2, pts2[i]);
                const double d1 = lineDistance(l1, pts1[i]);
                if (std::max(d1, d2) <= thrPx) {
                    maskF.push_back(1);
                    ++numF;
                } else {
                    maskF.push_back(0);
                }
            }
            if (numF >= 8) {
                g.F = F;
            } else {
                numF = 0;
                maskF.clear();
            }
        }
    } else if (!g.E.empty()) {
        // CALIBRATED：导出像素空间 F，供极线引导匹配与旧接口复用
        const cv::Mat Kinv = K.inv();
        g.F = Kinv.t() * g.E * Kinv;
    }
    st.lapF();
    st.dump(n, 0);

    // ── 配置判定 ────────────────────────────────────────────────────────────
    if (numE >= minInliers) {
        // cheirality 通过率：普通三维场景接近 1；平面 / 纯旋转下 E 分解出的
        // 位姿有歧义，点会前后各分一半，通过率掉到 0.5 左右
        const int cheiral = countMask(maskPose);
        const double frontRatio = numE > 0 ? static_cast<double>(cheiral) / numE : 0.0;
        g.numEInliers = numE;
        g.numHInliers = numH;
        g.poseFrontRatio = frontRatio;

        if (numH > numE * opt.planarDominanceRatio) {
            // 单应解释的内点更多：要么场景近似平面，要么近似纯旋转
            g.config = (frontRatio < 0.5) ? TWO_VIEW_PANORAMIC : TWO_VIEW_PLANAR;
            g.inlierMask = maskE; // 对应关系仍然成立（只是位姿不可靠）
        } else if (numH >= numE && frontRatio < 0.8) {
            // 噪声很小时 E 与 H 的内点数会打平（平面点同时满足两种模型），
            // 此时靠姿态歧义度区分：通过率明显低于 1 说明是平面场景
            g.config = (frontRatio < 0.5) ? TWO_VIEW_PANORAMIC : TWO_VIEW_PLANAR;
            g.inlierMask = maskE;
        } else {
            g.config = TWO_VIEW_CALIBRATED;
            g.inlierMask = maskPose.empty() ? maskE : maskPose;
            if (!R.empty() && !t.empty()) {
                g.R = R.clone();
                g.t = t.clone();
            }
        }
    } else if (numF >= minInliers) {
        if (numH > numF * opt.planarDominanceRatio) {
            g.config = TWO_VIEW_PLANAR;
            g.inlierMask = maskH;
        } else {
            g.config = TWO_VIEW_UNCALIBRATED;
            g.inlierMask = maskF;
        }
    } else if (numH >= minInliers) {
        g.config = TWO_VIEW_PLANAR;
        g.inlierMask = maskH;
    } else {
        g.config = TWO_VIEW_DEGENERATE;
        if (numE >= numF && numE >= numH) {
            g.inlierMask = maskE;
        } else if (numF >= numH) {
            g.inlierMask = maskF;
        } else {
            g.inlierMask = maskH;
        }
    }

    g.numInliers = countMask(g.inlierMask);
    g.inlierRatio = n > 0 ? static_cast<double>(g.numInliers) / static_cast<double>(n) : 0.0;

    if (g.numInliers < 8) {
        g.config = TWO_VIEW_DEGENERATE;
        g.inlierMask.clear();
        g.numInliers = 0;
        g.inlierRatio = 0.0;
    }
    return g;
}

int recoverPoseFromE(const cv::Mat &E,
                     const std::vector<cv::Point2d> &pts1,
                     const std::vector<cv::Point2d> &pts2,
                     const cv::Mat &K,
                     cv::Mat &R, cv::Mat &t,
                     std::vector<uchar> &mask)
{
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    if (E.empty() || !readK(K, fx, fy, cx, cy)) {
        return 0;
    }
    const size_t n = std::min(pts1.size(), pts2.size());
    if (n < 8) {
        return 0;
    }

    std::vector<cv::Point2d> n1, n2;
    normalizePoints(pts1, fx, fy, cx, cy, n, n1);
    normalizePoints(pts2, fx, fy, cx, cy, n, n2);

    try {
        const int inliers = static_cast<int>(
            cv::recoverPose(E, n1, n2, R, t, 1.0, cv::Point2d(0.0, 0.0), mask));
        return inliers;
    } catch (const cv::Exception &) {
        return 0;
    }
}

int guidedMatching(const std::vector<std::vector<cv::DMatch>> &knn,
                   const std::vector<cv::Point2d> &pts1,
                   const std::vector<cv::Point2d> &pts2,
                   const TwoViewGeometry &geom,
                   double maxErrorPx,
                   double ratioThreshold,
                   const std::vector<uchar> &usedQuery,
                   const std::vector<uchar> &usedTrain,
                   std::vector<cv::DMatch> &outMatches)
{
    if (knn.empty()) {
        return 0;
    }

    const bool useH = (geom.config == TWO_VIEW_PLANAR ||
                       geom.config == TWO_VIEW_PANORAMIC) &&
                      !geom.H.empty() && geom.H.type() == CV_64F;
    const bool useF = !geom.F.empty() && geom.F.type() == CV_64F;
    if (!useH && !useF) {
        return 0;
    }

    const int n2 = static_cast<int>(pts2.size());
    // 目标点占用表：在原有占用基础上，本轮新增的匹配也会占用，避免一靶多配
    std::vector<uchar> trainTaken = usedTrain;
    if (trainTaken.size() < static_cast<size_t>(n2)) {
        trainTaken.resize(n2, 0);
    }

    int added = 0;
    for (size_t i = 0; i < knn.size(); ++i) {
        if (i < usedQuery.size() && usedQuery[i]) {
            continue; // 该查询点已有匹配
        }
        if (i >= pts1.size()) {
            break;
        }

        const auto &cands = knn[i];
        if (cands.empty()) {
            continue;
        }

        // 预计算本查询点的几何约束（极线或单应投影点）
        cv::Vec3d line(0.0, 0.0, 0.0);
        cv::Point2d hTarget(0.0, 0.0);
        if (useH) {
            const double *h = geom.H.ptr<double>();
            const double w = h[6] * pts1[i].x + h[7] * pts1[i].y + h[8];
            if (std::fabs(w) < 1e-12) {
                continue;
            }
            hTarget = cv::Point2d((h[0] * pts1[i].x + h[1] * pts1[i].y + h[2]) / w,
                                  (h[3] * pts1[i].x + h[4] * pts1[i].y + h[5]) / w);
        } else {
            // 图像2 中的极线：l₂ = F · x₁（注意不是 Fᵀ·x₁，那是图像1 里的极线）
            const double *f = geom.F.ptr<double>();
            const double x = pts1[i].x, y = pts1[i].y;
            line = cv::Vec3d(f[0] * x + f[1] * y + f[2],
                             f[3] * x + f[4] * y + f[5],
                             f[6] * x + f[7] * y + f[8]);
        }

        double bestD = std::numeric_limits<double>::max();
        double secondD = std::numeric_limits<double>::max();
        int bestIdx = -1;
        int survivors = 0;

        for (const cv::DMatch &c : cands) {
            const int j = c.trainIdx;
            if (j < 0 || j >= n2 || trainTaken[j]) {
                continue;
            }
            const double err = useH
                                   ? std::hypot(hTarget.x - pts2[j].x, hTarget.y - pts2[j].y)
                                   : lineDistance(line, pts2[j]);
            if (err > maxErrorPx) {
                continue;
            }
            ++survivors;
            const double d = c.distance;
            if (d < bestD) {
                secondD = bestD;
                bestD = d;
                bestIdx = j;
            } else if (d < secondD) {
                secondD = d;
            }
        }

        if (bestIdx < 0) {
            continue;
        }
        // 幸存者内部仍有可比对象时做比率测试；只有唯一幸存者时直接接受
        // （极线带宽通常只有图像宽度的千分之一量级，假阳性概率已经很低）
        if (survivors >= 2 && bestD > ratioThreshold * secondD) {
            continue;
        }

        trainTaken[bestIdx] = 1;
        outMatches.emplace_back(static_cast<int>(i), bestIdx, 0, static_cast<float>(bestD));
        ++added;
    }
    return added;
}
