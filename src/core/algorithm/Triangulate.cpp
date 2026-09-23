/**
 * @file Triangulate.cpp
 * @brief 三角测量类实现 —— 增量式SFM稀疏重建核心算法
 *
 * 本文件实现了增量式SFM稀疏重建流程：
 * 1. 选取种子对，根据GRIC模型选择结果估计相对位姿
 * 2. 对种子对进行三角测量，生成初始三维点云
 * 3. 逐对三角测量 → PnP后方交会注册新相机 → 局部BA优化
 * 4. 全部完成后执行全局BA+自标定精化
 *
 * 光束法平差优化已剥离至 BundleAdjustment 类。
 */

#include "Triangulate.h"
#include "BundleAdjustment.h"

#include <QtMath>
#include <QVector>
#include <QDebug>
#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <unordered_map>

Triangulate *Triangulate::s_instance = nullptr;

Triangulate *Triangulate::instance()
{
    if (!s_instance) {
        s_instance = new Triangulate();
    }
    return s_instance;
}

Triangulate::Triangulate(QObject *parent)
    : QObject(parent)
{
}

cv::Mat Triangulate::estimateK(const std::vector<FeatureResult> &features)
{
    if (features.empty() || features[0].imageWidth < 1 || features[0].imageHeight < 1) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    return ::estimateK(features[0].imageWidth, features[0].imageHeight);
}

double Triangulate::computeImageFeatureDistributionScore(
    int imageId, const std::vector<FeatureResult> &features) const
{
    if (imageId < 0 || imageId >= static_cast<int>(features.size())) {
        return 0.0;
    }

    const FeatureResult &feature = features[imageId];
    const int width = feature.imageWidth;
    const int height = feature.imageHeight;
    if (width <= 0 || height <= 0 || feature.keypoints.empty()) {
        return 0.0;
    }

    // 三个网格层级：2x2、4x4、8x8，单格权重分别为 1、2、3
    const int gridCounts[3] = {2, 4, 8};
    const double cellWeights[3] = {1.0, 2.0, 3.0};

    double totalScore = 0.0;

    for (int lvl = 0; lvl < 3; ++lvl) {
        const int gridCount = gridCounts[lvl];
        const double cellWeight = cellWeights[lvl];

        std::vector<bool> occupied(gridCount * gridCount, false);

        const double cellW = static_cast<double>(width) / gridCount;
        const double cellH = static_cast<double>(height) / gridCount;

        for (const auto &kp : feature.keypoints) {
            int gx = static_cast<int>(kp.pt.x / cellW);
            int gy = static_cast<int>(kp.pt.y / cellH);
            gx = std::max(0, std::min(gridCount - 1, gx));
            gy = std::max(0, std::min(gridCount - 1, gy));
            occupied[gy * gridCount + gx] = true;
        }

        int occupiedCount = 0;
        for (bool occ : occupied) {
            if (occ) ++occupiedCount;
        }

        totalScore += occupiedCount * cellWeight;
    }

    return totalScore;
}

void Triangulate::triangulatePair(const MatchPair &pair,
                                   const cv::Mat &K1, const cv::Mat &K2,
                                   const cv::Mat &R1, const cv::Mat &t1,
                                   const cv::Mat &R2, const cv::Mat &t2,
                                   std::vector<cv::Point3d> &outWorldPoints,
                                   std::vector<bool> *outKeepMask)
{
    outWorldPoints.clear();

    cv::Mat T1 = (cv::Mat_<double>(3, 4) <<
        R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0, 0),
        R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1, 0),
        R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2, 0));

    cv::Mat T2 = (cv::Mat_<double>(3, 4) <<
        R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0, 0),
        R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1, 0),
        R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2, 0));

    std::vector<cv::Point2d> pts1 = pair.points1;
    std::vector<cv::Point2d> pts2 = pair.points2;

    if (pts1.empty()) {
        qDebug().noquote()
            << QString("[三角测量] pair #%1↔#%2: 无有效点").arg(pair.imageId1).arg(pair.imageId2);
        return;
    }

    cv::Mat ptsMat1(2, static_cast<int>(pts1.size()), CV_64F);
    cv::Mat ptsMat2(2, static_cast<int>(pts2.size()), CV_64F);
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Point2d n1 = pixel2cam(cv::Point2d(pts1[i].x, pts1[i].y), K1);
        cv::Point2d n2 = pixel2cam(cv::Point2d(pts2[i].x, pts2[i].y), K2);
        ptsMat1.at<double>(0, static_cast<int>(i)) = n1.x;
        ptsMat1.at<double>(1, static_cast<int>(i)) = n1.y;
        ptsMat2.at<double>(0, static_cast<int>(i)) = n2.x;
        ptsMat2.at<double>(1, static_cast<int>(i)) = n2.y;
    }

    cv::Mat pts4D;
    cv::triangulatePoints(T1, T2, ptsMat1, ptsMat2, pts4D);

    outWorldPoints.reserve(pts1.size());
    if (outKeepMask) {
        outKeepMask->resize(pts1.size(), false);
    }

    for (int i = 0; i < pts4D.cols; ++i) {
        double w = pts4D.at<double>(3, i);

        if (std::abs(w) < 1e-10) continue;

        double x = pts4D.at<double>(0, i) / w;
        double y = pts4D.at<double>(1, i) / w;
        double z = pts4D.at<double>(2, i) / w;

        if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
        if (std::isinf(x) || std::isinf(y) || std::isinf(z)) continue;
        if (z <= 0) continue;

        if (outKeepMask) (*outKeepMask)[i] = true;
        outWorldPoints.emplace_back(x, y, z);
    }

    qDebug().noquote()
        << QString("[三角测量] pair #%1↔#%2: 输入=%3 → 输出=%4")
               .arg(pair.imageId1 + 1).arg(pair.imageId2 + 1)
               .arg(pts1.size()).arg(outWorldPoints.size());
}

bool Triangulate::resection(const cv::Mat &K,
                             const std::vector<cv::KeyPoint> &kps,
                             const std::vector<cv::Point3d> &pts3d,
                             cv::Mat &rvec, cv::Mat &tvec)
{
    if (kps.size() != pts3d.size() || kps.size() < 4) {
        return false;
    }

    std::vector<cv::Point2d> imgPts;
    for (size_t i = 0; i < kps.size(); ++i) {
        imgPts.push_back(kps[i].pt);
    }

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    // 使用默认 ITERATIVE 方法（Levenberg-Marquardt），稳定性优于 SQPnP
    bool s = cv::solvePnPRansac(pts3d, imgPts, K, distCoeffs,
                                rvec, tvec, true, 1000, 8.0f, 0.99,
                                cv::noArray(), cv::SOLVEPNP_ITERATIVE);
    // 若失败则回退到 EPnP（对退化场景更鲁棒）
    if (!s) {
        s = cv::solvePnPRansac(pts3d, imgPts, K, distCoeffs,
                               rvec, tvec, false, 1000, 8.0f, 0.99,
                               cv::noArray(), cv::SOLVEPNP_EPNP);
    }
    return s;
}

/**
 * @brief 执行增量式三角测量重建（主入口）
 *
 * 完整流程：
 * 1. 使用 initResult 作为种子对，注册两张初始相机（世界坐标系 + 相对位姿）
 * 2. 在 ViewGraph 中找跨越已注册/未注册边界权重最大的前 4 条边，
 *    再基于特征点多分辨率网格分布得分从中确定 nextImage 和 bestReg
 * 3. 找到与 bestReg 有匹配对的 secondReg，获取两个匹配对
 * 4. 在 bestReg 的关键点上取两个匹配对的交集，得到三视图共视点
 * 5. 用 bestReg 和 secondReg 的已知位姿三角化 → 仅保留第三视图也可见的点
 * 6. 将三角化点追加到 outPoints3D（必须在局部BA之前，否则BA无法优化新点）
 * 7. 记录观测数据 → PnP 注册新相机 → 局部 BA 优化
 * 8. 重复步骤 2-7，直到无法注册新相机
 * 9. 全局 BA + 自标定精化
 */
bool Triangulate::triangulate(const std::vector<MatchPair> &matchPairs,
                               const std::vector<FeatureResult> &features,
                               const InitResult &initResult,
                               const ViewGraph &viewGraph,
                               std::vector<Point3D> &outPoints3D,
                               std::vector<CameraPose> &outCameraPoses,
                               bool fixIntrinsics)
{
    outPoints3D.clear();
    outCameraPoses.clear();
    m_lastEmittedPointCount = 0;

    int numImages = static_cast<int>(features.size());
    if (matchPairs.empty() || numImages < 3) {
        return false;
    }

    cv::Mat K = estimateK(features);

    // ── 初始化所有相机的内参和默认外参 ──
    outCameraPoses.resize(numImages);
    for (int c = 0; c < numImages; ++c) {
        outCameraPoses[c].imageId = c;
        outCameraPoses[c].imageWidth = features[c].imageWidth;
        outCameraPoses[c].imageHeight = features[c].imageHeight;
        outCameraPoses[c].K = K.clone();
        outCameraPoses[c].distortion = cv::Mat::zeros(5, 1, CV_64F);
    }

    // ── 种子对：第一张图像设为世界坐标系原点，第二张使用 initResult 的 R、t ──
    CameraPose &cam1 = outCameraPoses[initResult.imageId1];
    CameraPose &cam2 = outCameraPoses[initResult.imageId2];
    cam1.R = cv::Mat::eye(3, 3, CV_64F);
    cam1.t = cv::Mat::zeros(3, 1, CV_64F);
    cam1.registered = true;
    cam2.R = initResult.R.clone();
    cam2.t = initResult.t.clone();
    cam2.registered = true;

    // ── 已注册相机集合，用于贪心选择和三角化 ──
    std::set<int> registeredCameras;
    registeredCameras.insert(initResult.imageId1);
    registeredCameras.insert(initResult.imageId2);

    // ── BA 观测数据：(imageId, point3DIdx, pixelCoord) ──
    std::vector<std::tuple<int, int, cv::Point2d>> observations;

    // ── 辅助：按图像 ID 查找匹配对（无序，idA/idB 可互换） ──
    auto findMatchPair = [&](int idA, int idB) -> const MatchPair * {
        for (const auto &mp : matchPairs) {
            if ((mp.imageId1 == idA && mp.imageId2 == idB) ||
                (mp.imageId1 == idB && mp.imageId2 == idA)) {
                return &mp;
            }
        }
        return nullptr;
    };

    // ── 辅助：从匹配对中提取指定图像的关键点索引 ──
    //  MatchPair 中 imageId1 对应 queryIdx，imageId2 对应 trainIdx
    //  返回值：pair<该图像的kpt索引, 另一张图像的kpt索引>
    auto getKptIndices = [](const MatchPair &mp, int imageId) {
        std::vector<std::pair<int, int>> result;
        if (mp.imageId1 == imageId) {
            for (const auto &m : mp.matches)
                result.emplace_back(m.queryIdx, m.trainIdx);
        } else {
            for (const auto &m : mp.matches)
                result.emplace_back(m.trainIdx, m.queryIdx);
        }
        return result;
    };

    // ── 记录PnP失败的边 (nextImage, bestReg)，避免重复尝试 ──
    std::set<std::pair<int, int>> failedEdges;

    // ── 贪心选择：在 ViewGraph 中找到连接已注册子图与未注册节点的最大权重边 ──
    //  遍历所有边，若边的一端在已注册集合、另一端不在，则该边跨越注册边界
    //  取跨越边界权重最大的前 4 条边，再按特征点空间分布得分从中选出 nextImage
    //  跳过已失败的边，实现降级选择（匹配数第1→第2→第3...）
    auto selectNextEdge = [&]() -> std::pair<int, int> {
        // 收集所有跨越注册/未注册边界的边
        std::vector<ViewGraphEdge> boundaryEdges;
        for (const auto &edge : viewGraph.edges) {
            bool aRegistered = registeredCameras.count(edge.nodeA);
            bool bRegistered = registeredCameras.count(edge.nodeB);

            if (aRegistered == bRegistered) continue;

            int regNode = aRegistered ? edge.nodeA : edge.nodeB;
            int unregNode = aRegistered ? edge.nodeB : edge.nodeA;

            // 跳过已失败的边
            if (failedEdges.count({unregNode, regNode})) continue;

            boundaryEdges.push_back(edge);
        }

        if (boundaryEdges.empty()) {
            return {-1, -1};
        }

        // 按 matchCount 降序排序
        std::sort(boundaryEdges.begin(), boundaryEdges.end(),
                  [](const ViewGraphEdge &a, const ViewGraphEdge &b) {
                      return a.matchCount > b.matchCount;
                  });

        // 取前四条边（少于四则有多少取多少）
        const size_t topK = std::min<size_t>(4, boundaryEdges.size());

        // 按 nextImage 去重，保留权重最大的边（因已排序，首次出现即为最大）
        std::vector<ViewGraphEdge> candidates;
        std::set<int> seenUnreg;
        for (size_t i = 0; i < topK; ++i) {
            const auto &edge = boundaryEdges[i];
            bool aRegistered = registeredCameras.count(edge.nodeA);
            int unregNode = aRegistered ? edge.nodeB : edge.nodeA;
            if (seenUnreg.insert(unregNode).second) {
                candidates.push_back(edge);
            }
        }

        // 只有一个候选图像时直接返回，不计算特征分布权重
        if (candidates.size() == 1) {
            const auto &edge = candidates[0];
            bool aRegistered = registeredCameras.count(edge.nodeA);
            int regNode = aRegistered ? edge.nodeA : edge.nodeB;
            int unregNode = aRegistered ? edge.nodeB : edge.nodeA;
            return {unregNode, regNode};
        }

        // 基于特征点空间分布得分选择最佳候选图像
        double bestScore = -1.0;
        int bestNext = -1;
        int bestReg = -1;
        for (const auto &edge : candidates) {
            bool aRegistered = registeredCameras.count(edge.nodeA);
            int regNode = aRegistered ? edge.nodeA : edge.nodeB;
            int unregNode = aRegistered ? edge.nodeB : edge.nodeA;

            double score = computeImageFeatureDistributionScore(unregNode, features);
            if (score > bestScore) {
                bestScore = score;
                bestNext = unregNode;
                bestReg = regNode;
            }
        }

        return {bestNext, bestReg};
    };

    // ═══════════════════════════════════════════════════════════════════════
    //  增量注册主循环
    // ═══════════════════════════════════════════════════════════════════════
    while (true) {
        // ── 步骤 1：在 ViewGraph 中找跨越注册边界的最大权重前 4 条边，
        //           并按特征点分布得分选择最佳 nextImage ──
        //  返回 (nextImage, bestReg)：nextImage 是待注册图像，bestReg 是其最匹配的已注册图像
        auto [nextImage, bestReg] = selectNextEdge();
        if (nextImage < 0) {
            qDebug().noquote() << "[三角测量] 无更多可注册图像";
            break;
        }

        qDebug().noquote()
            << QString("[三角测量] 选择图像 #%1 (与已注册图像 #%2 特征分布得分最高)")
                   .arg(nextImage).arg(bestReg);

        // ── 步骤 2：在剩余已注册图像中，找到与 bestReg 有匹配对的 secondReg ──
        //  secondReg 用于和 bestReg 构成三角化基线，优先选择匹配数最多的
        int secondReg = -1;
        int secondRegMatchCount = 0;
        for (int regId : registeredCameras) {
            if (regId == bestReg) continue;
            const MatchPair *mp = findMatchPair(regId, bestReg);
            if (mp && mp->matchCount > secondRegMatchCount) {
                secondRegMatchCount = mp->matchCount;
                secondReg = regId;
            }
        }

        if (secondReg < 0) {
            qDebug().noquote()
                << QString("[三角测量] 图像 #%1 无法找到第二已注册图像构成三角化基线")
                       .arg(nextImage);
            break;
        }

        // ── 步骤 3：获取两个匹配对 ──
        //  mpReg: bestReg ↔ secondReg —— 用于两已注册相机间的三角化
        //  mpNew: bestReg ↔ nextImage —— 用于第三视图约束（将 nextImage 的关键点
        //          关联到 bestReg 的关键点，再通过 mpReg 关联到 secondReg）
        const MatchPair *mpReg = findMatchPair(bestReg, secondReg);
        const MatchPair *mpNew = findMatchPair(bestReg, nextImage);

        if (!mpReg || !mpNew) {
            qDebug().noquote()
                << QString("[三角测量] 图像 #%1 缺少必要匹配对").arg(nextImage);
            break;
        }

        // ── 步骤 4：在 bestReg 的关键点索引上取两个匹配对的交集 ──
        //  indicesReg: bestReg 的 kpt 索引 → secondReg 的 kpt 索引
        //  indicesNew: bestReg 的 kpt 索引 → nextImage 的 kpt 索引
        //  交集 = 同时出现在两个匹配对中的 bestReg 关键点 → 三视图共视点
        auto indicesReg = getKptIndices(*mpReg, bestReg);
        auto indicesNew = getKptIndices(*mpNew, bestReg);

        //  构建 bestReg kpt 索引 → secondReg kpt 索引 的查找表
        std::unordered_map<int, int> regLookup;
        regLookup.reserve(indicesReg.size());
        for (const auto &[kptBest, kptSecond] : indicesReg) {
            regLookup[kptBest] = kptSecond;
        }

        //  遍历 mpNew，在 regLookup 中查找共视点
        std::vector<cv::KeyPoint> kpsBest, kpsSecond, kpsNext;
        for (const auto &[kptBest, kptNext] : indicesNew) {
            auto it = regLookup.find(kptBest);
            if (it != regLookup.end()) {
                kpsBest.push_back(features[bestReg].keypoints[kptBest]);
                kpsSecond.push_back(features[secondReg].keypoints[it->second]);
                kpsNext.push_back(features[nextImage].keypoints[kptNext]);
            }
        }

        int commonCount = static_cast<int>(kpsBest.size());
        if (commonCount < 8) {
            qDebug().noquote()
                << QString("[三角测量] 图像 #%1 三视图共有点不足 (需要≥8, 实际=%2)")
                       .arg(nextImage).arg(commonCount);
            break;
        }

        qDebug().noquote()
            << QString("[三角测量] 注册图像 #%1: bestReg=#%2, secondReg=#%3, 三视图共有点=%4")
                   .arg(nextImage).arg(bestReg).arg(secondReg).arg(commonCount);

        // ── 步骤 5：用 bestReg 和 secondReg 的已知位姿进行三角化 ──
        cv::Mat R1 = outCameraPoses[bestReg].R.clone();
        cv::Mat t1 = outCameraPoses[bestReg].t.clone();
        cv::Mat R2 = outCameraPoses[secondReg].R.clone();
        cv::Mat t2 = outCameraPoses[secondReg].t.clone();

        //  组装临时 MatchPair 供 triangulatePair 使用
        MatchPair tempPair;
        tempPair.imageId1 = bestReg;
        tempPair.imageId2 = secondReg;
        for (int k = 0; k < commonCount; ++k) {
            tempPair.points1.push_back(kpsBest[k].pt);
            tempPair.points2.push_back(kpsSecond[k].pt);
            tempPair.matches.push_back(cv::DMatch(k, k, 0));
        }

        std::vector<cv::Point3d> pt3d;
        std::vector<bool> keepMask;
        triangulatePair(tempPair, K, K, R1, t1, R2, t2, pt3d, &keepMask);

        //  根据 keepMask 过滤掉重投影误差过大的点
        auto filterByMask = [&keepMask](auto &vec) {
            std::vector<cv::KeyPoint> filtered;
            for (size_t k = 0; k < vec.size() && k < keepMask.size(); ++k) {
                if (keepMask[k]) filtered.push_back(vec[k]);
            }
            vec = std::move(filtered);
        };
        if (!kpsBest.empty()) filterByMask(kpsBest);
        if (!kpsSecond.empty()) filterByMask(kpsSecond);
        if (!kpsNext.empty()) filterByMask(kpsNext);

        if (pt3d.empty()) {
            qDebug().noquote() << "[三角测量] 三角化失败，无有效三维点";
            break;
        }

        // ── 步骤 6：将三角化得到的三维点追加到输出点云（必须在局部BA之前） ──
        //  局部BA需要从 outPoints3D 构建 ptsGlobal，若点未追加则 BA 无法优化新点
        //  记录追加前的索引，PnP失败时回滚
        int basePointIdx = static_cast<int>(outPoints3D.size());
        int baseObservationIdx = static_cast<int>(observations.size());
        const cv::Mat &bestImg = features[bestReg].imageData;
        for (size_t i = 0; i < pt3d.size(); ++i) {
            Point3D pt;
            pt.id = static_cast<int>(outPoints3D.size());
            pt.position = pt3d[i];

            // 从 bestReg 图像的关键点位置取颜色
            pt.color = cv::Vec3b(128, 128, 128);
            if (i < kpsBest.size() && !bestImg.empty()) {
                int px = static_cast<int>(kpsBest[i].pt.x + 0.5);
                int py = static_cast<int>(kpsBest[i].pt.y + 0.5);
                if (py >= 0 && py < bestImg.rows && px >= 0 && px < bestImg.cols) {
                    pt.color = bestImg.at<cv::Vec3b>(py, px);
                }
            }

            outPoints3D.push_back(pt);
        }

        // ── 步骤 7：记录观测数据 ──
        //  每个三角化点在三张图像上的像素观测都加入 observations，供 BA 使用
        for (size_t k = 0; k < pt3d.size(); ++k) {
            int ptIdx = basePointIdx + static_cast<int>(k);
            if (k < kpsBest.size()) {
                observations.emplace_back(bestReg, ptIdx,
                    cv::Point2d(kpsBest[k].pt.x, kpsBest[k].pt.y));
            }
            if (k < kpsSecond.size()) {
                observations.emplace_back(secondReg, ptIdx,
                    cv::Point2d(kpsSecond[k].pt.x, kpsSecond[k].pt.y));
            }
            if (k < kpsNext.size()) {
                observations.emplace_back(nextImage, ptIdx,
                    cv::Point2d(kpsNext[k].pt.x, kpsNext[k].pt.y));
            }
        }

        // ── 步骤 8：PnP 后方交会注册新相机 ──
        //  使用第三视图的 2D-3D 对应关系求解 nextImage 的位姿
        // cv::Mat rvec, tvec;
        if (resection(outCameraPoses[nextImage].K, kpsNext, pt3d, R1, t1)) {
            cv::Mat R_new;
            cv::Rodrigues(R1, R_new);

            outCameraPoses[nextImage].R = R_new.clone();
            outCameraPoses[nextImage].t = t1.clone();
            outCameraPoses[nextImage].registered = true;
            registeredCameras.insert(nextImage);

            // PnP成功：清除该图像的所有失败记录
            for (auto it = failedEdges.begin(); it != failedEdges.end(); ) {
                if (it->first == nextImage) {
                    it = failedEdges.erase(it);
                } else {
                    ++it;
                }
            }

            qDebug().noquote()
                << QString("[三角测量] PnP 注册相机 #%1 成功").arg(nextImage);

            // ── 步骤 9：局部 BA ──
            //  仅优化新注册相机 nextImage 的外参 + basePointIdx 之后新增的三维点
            //  已有相机和旧三维点保持固定，保证增量重建的稳定性
            {
                std::vector<std::array<double, 6>> cameras(numImages);
                for (int c = 0; c < numImages; ++c) {
                    cv::Mat rvec_c;
                    cv::Mat Rc = outCameraPoses[c].R.empty()
                                     ? cv::Mat::eye(3, 3, CV_64F) : outCameraPoses[c].R;
                    cv::Mat tc = outCameraPoses[c].t.empty()
                                     ? cv::Mat::zeros(3, 1, CV_64F) : outCameraPoses[c].t;
                    cv::Rodrigues(Rc, rvec_c);
                    cameras[c] = {rvec_c.at<double>(0), rvec_c.at<double>(1), rvec_c.at<double>(2),
                                  tc.at<double>(0), tc.at<double>(1), tc.at<double>(2)};
                }

                std::vector<std::array<double, 3>> ptsGlobal;
                ptsGlobal.reserve(outPoints3D.size());
                for (const auto &p : outPoints3D) {
                    ptsGlobal.push_back({p.position.x, p.position.y, p.position.z});
                }

                qDebug().noquote()
                    << QString("[三角测量] 局部BA: 相机#%1, %2点")
                           .arg(nextImage).arg(ptsGlobal.size());

                BundleAdjustment::instance()->bundleAdjustment(
                    cameras, ptsGlobal, observations,
                    K.at<double>(0, 0), K.at<double>(1, 1),
                    K.at<double>(0, 2), K.at<double>(1, 2),
                    nextImage, basePointIdx);

                //  回写局部 BA 优化后的新相机位姿
                cv::Mat rvec_updated = (cv::Mat_<double>(3, 1) <<
                    cameras[nextImage][0], cameras[nextImage][1], cameras[nextImage][2]);
                cv::Mat Rj;
                cv::Rodrigues(rvec_updated, Rj);
                outCameraPoses[nextImage].R = Rj.clone();
                outCameraPoses[nextImage].t = (cv::Mat_<double>(3, 1) <<
                    cameras[nextImage][3], cameras[nextImage][4], cameras[nextImage][5]);

                //  回写局部 BA 优化后的三维点坐标
                for (size_t k = 0; k < ptsGlobal.size(); ++k) {
                    outPoints3D[k].position = cv::Point3d(
                        ptsGlobal[k][0], ptsGlobal[k][1], ptsGlobal[k][2]);
                }
            }
        } else {
            // PnP失败：回滚步骤6-7追加的点和观测，记录失败的边
            outPoints3D.resize(basePointIdx);
            observations.resize(baseObservationIdx);
            failedEdges.insert({nextImage, bestReg});
            qDebug().noquote()
                << QString("[三角测量] PnP 注册相机 #%1 失败 (使用已注册图像 #%2)，尝试次优匹配")
                       .arg(nextImage).arg(bestReg);
            continue;
        }

        // ── 实时推送增量更新（只发新增点） ──
        {
            QVector<float> pts, cls;
            int startIdx = m_lastEmittedPointCount;
            pts.reserve(static_cast<int>(outPoints3D.size() - startIdx) * 3);
            cls.reserve(static_cast<int>(outPoints3D.size() - startIdx) * 3);
            for (size_t i = startIdx; i < outPoints3D.size(); ++i) {
                const auto &p = outPoints3D[i];
                pts << static_cast<float>(p.position.x)
                    << static_cast<float>(p.position.y)
                    << static_cast<float>(p.position.z);
                cls << p.color[2] / 255.0f
                    << p.color[1] / 255.0f
                    << p.color[0] / 255.0f;
            }
            m_lastEmittedPointCount = static_cast<int>(outPoints3D.size());

            QVector<float> camData;
            camData.reserve(registeredCameras.size() * 6);
            for (int camId : registeredCameras) {
                cv::Mat C = -outCameraPoses[camId].R.t() * outCameraPoses[camId].t;
                cv::Mat fwd = outCameraPoses[camId].R.row(2).t();
                camData << static_cast<float>(C.at<double>(0))
                        << static_cast<float>(C.at<double>(1))
                        << static_cast<float>(C.at<double>(2))
                        << static_cast<float>(fwd.at<double>(0))
                        << static_cast<float>(fwd.at<double>(1))
                        << static_cast<float>(fwd.at<double>(2));
            }
            emit incrementalUpdate(pts, cls, camData, startIdx == 0);
        }
    }

    if (outPoints3D.empty()) {
        qDebug().noquote() << "[三角测量] 重建失败: 无有效三维点";
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  全局 BA + 自标定精化
    //  联合优化所有已注册相机的外参、内参(fx,fy,cx,cy)和全部三维点坐标
    //  第一张图像的外参固定不动（作为世界坐标系参考）
    //  使用 Huber 鲁棒核抑制外点干扰
    // ═══════════════════════════════════════════════════════════════════════
    {
        std::vector<std::array<double, 6>> cameras(numImages);
        for (int c = 0; c < numImages; ++c) {
            cv::Mat rvec;
            cv::Mat R = outCameraPoses[c].R.empty()
                            ? cv::Mat::eye(3, 3, CV_64F) : outCameraPoses[c].R;
            cv::Mat t = outCameraPoses[c].t.empty()
                            ? cv::Mat::zeros(3, 1, CV_64F) : outCameraPoses[c].t;
            cv::Rodrigues(R, rvec);
            cameras[c] = {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                          t.at<double>(0), t.at<double>(1), t.at<double>(2)};
        }

        std::vector<std::array<double, 3>> ptsGlobal;
        ptsGlobal.reserve(outPoints3D.size());
        for (const auto &p : outPoints3D) {
            ptsGlobal.push_back({p.position.x, p.position.y, p.position.z});
        }

        qDebug().noquote()
            << QString("[三角测量] 全局BA+自标定: %1 相机, %2 点")
                   .arg(registeredCameras.size())
                   .arg(outPoints3D.size());

        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);

        BundleAdjustment::instance()->globalBundleAdjustment(
            cameras, ptsGlobal, observations,
            fx, fy, cx, cy, fixIntrinsics);

        //  回写全局 BA 优化后的所有相机位姿
        for (int c = 0; c < numImages; ++c) {
            cv::Mat rvec = (cv::Mat_<double>(3, 1) <<
                                cameras[c][0], cameras[c][1], cameras[c][2]);
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            outCameraPoses[c].R = R.clone();
            outCameraPoses[c].t = (cv::Mat_<double>(3, 1) <<
                                       cameras[c][3], cameras[c][4], cameras[c][5]);
        }

        //  回写全局 BA 优化后的三维点坐标
        for (size_t k = 0; k < ptsGlobal.size(); ++k) {
            outPoints3D[k].position = cv::Point3d(
                ptsGlobal[k][0], ptsGlobal[k][1], ptsGlobal[k][2]);
        }

        //  回写自标定精化后的内参
        K.at<double>(0, 0) = fx;
        K.at<double>(1, 1) = fy;
        K.at<double>(0, 2) = cx;
        K.at<double>(1, 2) = cy;
        for (int c = 0; c < numImages; ++c) {
            outCameraPoses[c].K = K.clone();
        }

        // ── 全局BA完成后推送全量更新（BA修改了所有点坐标） ──
        {
            QVector<float> pts, cls;
            pts.reserve(static_cast<int>(outPoints3D.size()) * 3);
            cls.reserve(static_cast<int>(outPoints3D.size()) * 3);
            for (const auto &p : outPoints3D) {
                pts << static_cast<float>(p.position.x)
                    << static_cast<float>(p.position.y)
                    << static_cast<float>(p.position.z);
                cls << p.color[2] / 255.0f
                    << p.color[1] / 255.0f
                    << p.color[0] / 255.0f;
            }
            QVector<float> camData;
            camData.reserve(registeredCameras.size() * 6);
            for (int camId : registeredCameras) {
                cv::Mat C = -outCameraPoses[camId].R.t() * outCameraPoses[camId].t;
                cv::Mat fwd = outCameraPoses[camId].R.row(2).t();
                camData << static_cast<float>(C.at<double>(0))
                        << static_cast<float>(C.at<double>(1))
                        << static_cast<float>(C.at<double>(2))
                        << static_cast<float>(fwd.at<double>(0))
                        << static_cast<float>(fwd.at<double>(1))
                        << static_cast<float>(fwd.at<double>(2));
            }
            m_lastEmittedPointCount = static_cast<int>(outPoints3D.size());
            emit incrementalUpdate(pts, cls, camData, true);
        }
    }

    qDebug().noquote()
        << QString("[三角测量] 重建完成: 已注册相机=%1, 三维点总数=%2")
               .arg(registeredCameras.size()).arg(outPoints3D.size());

    return true;
}