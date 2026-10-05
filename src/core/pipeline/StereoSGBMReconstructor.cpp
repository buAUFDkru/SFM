#include "StereoSGBMReconstructor.h"
#include "DenseReconstructionFactory.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>

namespace {

DenseStereoConfig buildConfig(const QMap<QString, QVariant> &params, const QString &projectPath)
{
    DenseStereoConfig config;

    QString rectifiedDir = !projectPath.isEmpty()
                               ? projectPath + "/image"
                               : QCoreApplication::applicationDirPath() + "/output/image";
    config.saveRectifiedDir = rectifiedDir.toStdString();
    config.maxDepth = params.value("maxDepth", 50.0).toDouble();
    config.useWLSFilter = params.value("useWLSFilter", true).toBool();
    config.minDisparity = params.value("minDisparity", 0).toInt();
    config.numDisparities = params.value("numDisparities", 160).toInt();
    config.blockSize = params.value("blockSize", 3).toInt();
    config.useHHMode = params.value("useHHMode", true).toBool();
    return config;
}

} // namespace

// 注：StereoSGBM 由 DenseReconstructionFactory::registerBuiltinMethods() 显式注册
//（静态库链接时，此处的静态注册对象可能不被拉入）。

// ═══════════════════════════════════════════════════════════════
// 接口实现
// ═══════════════════════════════════════════════════════════════

void StereoSGBMReconstructor::setConfig(const QMap<QString, QVariant> &params)
{
    m_params = params;
    m_config = buildConfig(params, QString());
}

DenseReconstructionOutput StereoSGBMReconstructor::reconstruct(const DenseReconstructionInput &input)
{
    DenseReconstructionOutput output;
    output.denseCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (input.cameraPoses.empty() || input.imageFiles.isEmpty()) {
        output.errorMessage = QStringLiteral("输入数据为空");
        return output;
    }

    // 用项目路径重建配置（saveRectifiedDir 等需要项目路径）
    m_config = buildConfig(m_params, input.projectPath);

    // 注：setLogCallback 已将回调转发给 m_rectifier，此处无需重复设置

    // 选图策略
    std::vector<StereoPairInfo> stereoPairs =
        selectStereoPairs(input.cameraPoses, input.matchPairs,
                          static_cast<int>(input.imageFiles.size()));

    log(QString("[密集匹配] 选中 %1 对立体匹配对").arg(stereoPairs.size()));

    // 填充内参、位姿和图像路径（不在这里加载图像，避免一次性把所有图读进内存）
    const cv::Mat K = input.cameraMatrix.empty()
                          ? cv::Mat()
                          : input.cameraMatrix.clone();

    for (auto &pair : stereoPairs) {
        if (pair.imageId1 >= 0 && pair.imageId1 < input.imageFiles.size()) {
            pair.imagePath1 = input.imageFiles[pair.imageId1];
        }
        if (pair.imageId2 >= 0 && pair.imageId2 < input.imageFiles.size()) {
            pair.imagePath2 = input.imageFiles[pair.imageId2];
        }

        if (pair.imageId1 >= 0 && pair.imageId1 < static_cast<int>(input.cameraPoses.size())) {
            const auto &pose1 = input.cameraPoses[pair.imageId1];
            pair.K1 = pose1.K.empty() ? K.clone() : pose1.K.clone();
            pair.R1 = pose1.R.clone();
            pair.t1 = pose1.t.clone();
        }
        if (pair.imageId2 >= 0 && pair.imageId2 < static_cast<int>(input.cameraPoses.size())) {
            const auto &pose2 = input.cameraPoses[pair.imageId2];
            pair.K2 = pose2.K.empty() ? K.clone() : pose2.K.clone();
            pair.R2 = pose2.R.clone();
            pair.t2 = pose2.t.clone();
        }
    }

    // 移除路径无效的对
    stereoPairs.erase(
        std::remove_if(stereoPairs.begin(), stereoPairs.end(),
            [](const StereoPairInfo &p) { return p.imagePath1.isEmpty() || p.imagePath2.isEmpty(); }),
        stereoPairs.end());

    log(QString("[密集匹配] 有效对数: %1（图像在 processPair 内流式加载）").arg(stereoPairs.size()));
    if (stereoPairs.empty()) {
        output.errorMessage = QStringLiteral("没有有效的立体匹配对");
        return output;
    }

    // 执行立体匹配
    output.stereoResults = processAllPairs(stereoPairs, input.matchPairs);
    log(QString("[密集匹配] 全部处理完成, 结果数=%1").arg(output.stereoResults.size()));

    // 从深度图反投影生成密集点云
    for (size_t i = 0; i < output.stereoResults.size(); ++i) {
        const auto &result = output.stereoResults[i];
        if (result.depth.empty() || result.K_rect.empty()) {
            continue;
        }

        int refId = result.imageId1;
        if (refId < 0 || refId >= static_cast<int>(input.cameraPoses.size())) {
            continue;
        }
        const auto &pose = input.cameraPoses[refId];
        if (!pose.registered) {
            continue;
        }

        // 位姿约定: P_cam = pose.R * P_world + pose.t (world→camera)
        // 校正: P_rect = R1_rect * P_cam (原始相机→校正相机)
        // 反向: P_cam_orig = R1_rect^T * P_rect
        //        P_world = pose.R^T * (P_cam_orig - pose.t)
        //               = pose.R^T * R1_rect^T * P_rect - pose.R^T * pose.t
        cv::Mat R_world;
        if (!result.R1_rect.empty()) {
            R_world = (result.R1_rect * pose.R).t();   // pose.R^T * R1_rect^T
        } else {
            R_world = pose.R.t();                       // pose.R^T
        }
        cv::Mat t_world = -pose.R.t() * pose.t;        // -pose.R^T * pose.t

        const double fx = result.K_rect.at<double>(0, 0);
        const double fy = result.K_rect.at<double>(1, 1);
        const double cx = result.K_rect.at<double>(0, 2);
        const double cy = result.K_rect.at<double>(1, 2);

        // 采样步长（避免点云过密）
        constexpr int step = 2;

        const cv::Mat &depthMap = result.depth;
        const cv::Mat &colorImg = result.rectifiedImage1;

        for (int y = 0; y < depthMap.rows; y += step) {
            for (int x = 0; x < depthMap.cols; x += step) {
                float d = depthMap.at<float>(y, x);
                if (d <= 0.0f || !std::isfinite(d)) {
                    continue;
                }

                double Xc = (x - cx) * d / fx;
                double Yc = (y - cy) * d / fy;
                double Zc = d;

                double Xw = R_world.at<double>(0, 0) * Xc + R_world.at<double>(0, 1) * Yc + R_world.at<double>(0, 2) * Zc + t_world.at<double>(0);
                double Yw = R_world.at<double>(1, 0) * Xc + R_world.at<double>(1, 1) * Yc + R_world.at<double>(1, 2) * Zc + t_world.at<double>(1);
                double Zw = R_world.at<double>(2, 0) * Xc + R_world.at<double>(2, 1) * Yc + R_world.at<double>(2, 2) * Zc + t_world.at<double>(2);

                uint8_t r = 128, g = 128, b = 128;
                if (!colorImg.empty() && colorImg.type() == CV_8UC3) {
                    cv::Vec3b c = colorImg.at<cv::Vec3b>(y, x);
                    b = c[0]; g = c[1]; r = c[2];
                }

                if (r + g + b < 15) {
                    continue;
                }

                pcl::PointXYZRGB pt;
                pt.x = static_cast<float>(Xw);
                pt.y = static_cast<float>(Yw);
                pt.z = static_cast<float>(Zw);
                pt.r = r; pt.g = g; pt.b = b;
                output.denseCloud->push_back(pt);
            }
        }
    }

    output.success = !output.denseCloud->empty();
    if (!output.success && output.errorMessage.isEmpty()) {
        output.errorMessage = QStringLiteral("未生成有效密集点云");
    }
    return output;
}

// ═══════════════════════════════════════════════════════════════
// 第一步：选图策略
//
// 基于稀疏点云重叠度和基线角度，为每张视图选择最优的匹配伙伴。
// 评分 = 重叠率 * 0.6 + 角度得分 * 0.4
// 角度得分使用高斯核，峰值在15度（平衡基线长度和匹配难度）
// ═══════════════════════════════════════════════════════════════

std::vector<StereoPairInfo> StereoSGBMReconstructor::selectStereoPairs(
    const std::vector<CameraPose> &cameraPoses,
    const std::vector<MatchPair> &matchPairs,
    int /*numImages*/)
{
    struct PairScore {
        int id1, id2;
        int numShared;
        double overlapRatio;
        double angle;
        double score;
    };

    // ── 1. 统计每张图像的特征点数 ──
    std::unordered_map<int, int> imageFeatureCount;
    for (const auto &mp : matchPairs) {
        imageFeatureCount[mp.imageId1] += mp.matchCount;
        imageFeatureCount[mp.imageId2] += mp.matchCount;
    }

    // ── 2. 计算每对图像的评分 ──
    std::vector<PairScore> pairScores;
    for (const auto &mp : matchPairs) {
        int id1 = std::min(mp.imageId1, mp.imageId2);
        int id2 = std::max(mp.imageId1, mp.imageId2);
        int numShared = mp.matchCount;

        int total1 = imageFeatureCount[id1];
        int total2 = imageFeatureCount[id2];
        int unionSize = total1 + total2 - numShared;
        double overlapRatio = (unionSize > 0) ? static_cast<double>(numShared) / unionSize : 0.0;

        double angle = 0.0;
        if (id1 < static_cast<int>(cameraPoses.size()) &&
            id2 < static_cast<int>(cameraPoses.size()) &&
            cameraPoses[id1].registered && cameraPoses[id2].registered) {
            angle = StereoRectifier::computeViewAngle(cameraPoses[id1].R, cameraPoses[id2].R);
        }

        double angleScore = 0.0;
        if (angle >= m_config.minAngleDeg && angle <= m_config.maxAngleDeg) {
            angleScore = std::exp(-std::pow(angle - 15.0, 2) / 200.0);
        }

        double score = overlapRatio * 0.6 + angleScore * 0.4;

        if (overlapRatio >= m_config.minOverlapRatio) {
            pairScores.push_back({id1, id2, numShared, overlapRatio, angle, score});
        }
    }

    // ── 3. 按综合评分降序排序 ──
    std::sort(pairScores.begin(), pairScores.end(),
              [](const PairScore &a, const PairScore &b) { return a.score > b.score; });

    // ── 4. 贪心选择 ──
    std::unordered_map<int, int> viewPairCount;
    std::unordered_set<int> selectedPairKeys;
    std::vector<StereoPairInfo> selected;

    for (const auto &ps : pairScores) {
        if (viewPairCount[ps.id1] >= m_config.targetPairsPerView &&
            viewPairCount[ps.id2] >= m_config.targetPairsPerView) {
            continue;
        }

        int key = ps.id1 * 100000 + ps.id2;
        if (selectedPairKeys.count(key)) continue;

        if (ps.id1 >= static_cast<int>(cameraPoses.size()) ||
            ps.id2 >= static_cast<int>(cameraPoses.size())) continue;
        const auto &pose1 = cameraPoses[ps.id1];
        const auto &pose2 = cameraPoses[ps.id2];
        if (!pose1.registered || !pose2.registered) continue;

        StereoPairInfo info;
        info.imageId1 = ps.id1;
        info.imageId2 = ps.id2;
        info.K1 = pose1.K.clone();
        info.K2 = pose2.K.clone();
        info.R1 = pose1.R.clone();
        info.t1 = pose1.t.clone();
        info.R2 = pose2.R.clone();
        info.t2 = pose2.t.clone();
        info.image1 = cv::Mat();
        info.image2 = cv::Mat();

        selected.push_back(info);
        selectedPairKeys.insert(key);
        viewPairCount[ps.id1]++;
        viewPairCount[ps.id2]++;
    }

    return selected;
}

// ═══════════════════════════════════════════════════════════════
// SGBM立体匹配 + WLS滤波
//
// SGBM (Semi-Global Block Matching) 是一种半全局立体匹配算法：
//   - 在每个像素处计算匹配代价（Census/SAD）
//   - 沿8个方向进行动态规划，累积平滑代价
//   - HH模式在全分辨率上执行，精度更高
//
// WLS (Weighted Least Squares) 后滤波：
//   - 利用左右一致性检查和图像边缘信息
//   - Lambda控制平滑强度，Sigma控制边缘敏感度
// ═══════════════════════════════════════════════════════════════

bool StereoSGBMReconstructor::stereoMatch(StereoPairResult &result,
                                          const std::vector<cv::Point2d> &matchedPoints1,
                                          const std::vector<cv::Point2d> &matchedPoints2)
{
    if (result.rectifiedImage1.empty() || result.rectifiedImage2.empty()) {
        return false;
    }

    // ── 转灰度 ──
    cv::Mat grayL, grayR;
    if (result.rectifiedImage1.channels() == 3)
        cv::cvtColor(result.rectifiedImage1, grayL, cv::COLOR_BGR2GRAY);
    else
        grayL = result.rectifiedImage1;
    if (result.rectifiedImage2.channels() == 3)
        cv::cvtColor(result.rectifiedImage2, grayR, cv::COLOR_BGR2GRAY);
    else
        grayR = result.rectifiedImage2;

    // ── 图像缩放（防止高分辨率图像导致SGBM内存爆炸） ──
    const int kMaxWidth = 2000;
    double scale = 1.0;
    if (grayL.cols > kMaxWidth) {
        scale = static_cast<double>(kMaxWidth) / grayL.cols;
    }
    cv::Mat workL, workR;
    if (scale < 1.0) {
        cv::resize(grayL, workL, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(grayR, workR, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        workL = grayL;
        workR = grayR;
    }

    // ── 确定视差搜索范围 ──
    int minDisp = m_config.minDisparity;
    int numDisp = m_config.numDisparities;

    if (m_config.autoDisparityRange && !matchedPoints1.empty() && !matchedPoints2.empty()) {
        estimateDisparityRange(matchedPoints1, matchedPoints2, minDisp, numDisp);
    }

    if (scale < 1.0) {
        minDisp = static_cast<int>(std::round(minDisp * scale));
        numDisp = static_cast<int>(std::round(numDisp * scale));
    }

    numDisp = ((numDisp + 15) / 16) * 16;
    if (numDisp <= 0) numDisp = 16;

    int maxAllowedDisp = workL.cols - minDisp - 1;
    if (maxAllowedDisp < 16) maxAllowedDisp = 16;
    if (numDisp > maxAllowedDisp) {
        numDisp = (maxAllowedDisp / 16) * 16;
        if (numDisp <= 0) numDisp = 16;
        log(QString("[密集匹配]     视差范围被限制到 %1 (图像宽度=%2, 最小视差=%3)")
                .arg(numDisp).arg(workL.cols).arg(minDisp));
    }

    if (scale < 1.0 && result.baseline > 0.0) {
        double fx_scaled = result.K_rect.at<double>(0, 0) * scale;
        double baseline = result.baseline;
        int minRequiredNumDisp = static_cast<int>(std::ceil(fx_scaled * baseline / 4.0));
        minRequiredNumDisp = ((minRequiredNumDisp + 15) / 16) * 16;
        if (minRequiredNumDisp > maxAllowedDisp) {
            minRequiredNumDisp = maxAllowedDisp;
        }
        if (numDisp < minRequiredNumDisp) {
            log(QString("[密集匹配]     高分辨率图像自动增大视差范围: %1 -> %2 (fx*base=%3)")
                    .arg(numDisp).arg(minRequiredNumDisp)
                    .arg(fx_scaled * baseline, 0, 'f', 1));
            numDisp = minRequiredNumDisp;
        }
    }

    log(QString("[密集匹配]     SGBM参数: 工作尺寸=%1x%2, 最小视差=%3, 视差范围=%4")
            .arg(workL.cols).arg(workL.rows).arg(minDisp).arg(numDisp));

    // ── 计算SGBM惩罚参数 ──
    int channels = workL.channels();
    int bs = m_config.blockSize;
    int P1 = m_config.P1Scale * channels * bs * bs;
    int P2 = m_config.P2Scale * channels * bs * bs;
    int mode = m_config.useHHMode ? cv::StereoSGBM::MODE_HH : cv::StereoSGBM::MODE_SGBM;

    // ── 左→右 SGBM匹配 ──
    cv::Ptr<cv::StereoSGBM> sgbmL = cv::StereoSGBM::create(
        minDisp, numDisp, bs, P1, P2,
        m_config.disp12MaxDiff,
        m_config.preFilterCap,
        m_config.uniquenessRatio,
        m_config.speckleWindowSize,
        m_config.speckleRange,
        mode);

    cv::Mat dispL;
    sgbmL->compute(workL, workR, dispL);

    cv::Mat dispL32F;
    dispL.convertTo(dispL32F, CV_32FC1, 1.0 / 16.0);

    // ── 不使用WLS滤波，直接输出 ──
    if (!m_config.useWLSFilter) {
        result.disparity = dispL32F;
        if (scale < 1.0) {
            result.K_rect = result.K_rect * scale;
            result.K_rect.at<double>(2, 2) = 1.0;
            if (!result.Q.empty()) {
                result.Q.at<double>(0, 3) *= scale;
                result.Q.at<double>(1, 3) *= scale;
                result.Q.at<double>(2, 3) *= scale;
            }
            cv::resize(result.rectifiedImage1, result.rectifiedImage1, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        return true;
    }

    // ── 右→左匹配（用于WLS滤波的左右一致性检查） ──
    cv::Ptr<cv::StereoMatcher> sgbmR = cv::ximgproc::createRightMatcher(sgbmL);
    cv::Mat dispR;
    sgbmR->compute(workR, workL, dispR);
    cv::Mat dispR32F;
    dispR.convertTo(dispR32F, CV_32FC1, 1.0 / 16.0);

    // ── WLS滤波 ──
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter =
        cv::ximgproc::createDisparityWLSFilter(sgbmL);
    wlsFilter->setLambda(m_config.wlsLambda);
    wlsFilter->setSigmaColor(m_config.wlsSigmaColor);

    cv::Mat filteredDisp;
    cv::Mat workColor;
    if (scale < 1.0) {
        cv::resize(result.rectifiedImage1, workColor, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        workColor = result.rectifiedImage1;
    }
    wlsFilter->filter(dispL32F, workColor, filteredDisp, dispR32F);

    // ── 后处理：移除无效视差 ──
    filteredDisp.setTo(0, filteredDisp < 0);
    filteredDisp.setTo(0, filteredDisp >= numDisp - 1);

    result.disparity = filteredDisp;
    if (scale < 1.0) {
        result.K_rect = result.K_rect * scale;
        result.K_rect.at<double>(2, 2) = 1.0;
        if (!result.Q.empty()) {
            result.Q.at<double>(0, 3) *= scale;
            result.Q.at<double>(1, 3) *= scale;
            result.Q.at<double>(2, 3) *= scale;
        }
        cv::resize(result.rectifiedImage1, result.rectifiedImage1, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════
// 视差转深度
//   depth = fx * baseline / disparity
// ═══════════════════════════════════════════════════════════════

bool StereoSGBMReconstructor::disparityToDepth(StereoPairResult &result)
{
    if (result.disparity.empty() || result.K_rect.empty()) {
        return false;
    }

    double fx = result.K_rect.at<double>(0, 0);
    double baseline = result.baseline;

    if (baseline <= 0.0 || fx <= 0.0) {
        return false;
    }

    result.depth = cv::Mat::zeros(result.disparity.size(), CV_32FC1);

    double minDepthVal = std::numeric_limits<double>::max();
    double maxDepthVal = 0.0;
    double sumDepth = 0.0;
    int validCount = 0;

    for (int y = 0; y < result.disparity.rows; ++y) {
        const float *dispPtr = result.disparity.ptr<float>(y);
        float *depthPtr = result.depth.ptr<float>(y);
        for (int x = 0; x < result.disparity.cols; ++x) {
            float d = dispPtr[x];
            if (d > 1.0f) {
                float depth = static_cast<float>(fx * baseline / d);
                if (depth >= static_cast<float>(m_config.minDepth) &&
                    depth <= static_cast<float>(m_config.maxDepth)) {
                    depthPtr[x] = depth;
                    minDepthVal = std::min(minDepthVal, static_cast<double>(depth));
                    maxDepthVal = std::max(maxDepthVal, static_cast<double>(depth));
                    sumDepth += depth;
                    validCount++;
                }
            }
        }
    }

    if (validCount > 0) {
        log(QString("[密集匹配] 深度统计: 有效像素=%1, 范围=[%2, %3], 均值=%4, fx*base=%5")
                .arg(validCount)
                .arg(minDepthVal, 0, 'f', 2).arg(maxDepthVal, 0, 'f', 2)
                .arg(sumDepth / validCount, 0, 'f', 2)
                .arg(fx * baseline, 0, 'f', 2));
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════
// 批量处理所有立体匹配对
// ═══════════════════════════════════════════════════════════════

std::vector<StereoPairResult> StereoSGBMReconstructor::processAllPairs(
    const std::vector<StereoPairInfo> &pairs,
    const std::vector<MatchPair> &matchPairs)
{
    std::vector<StereoPairResult> results;
    results.reserve(pairs.size());
    m_rectifyPairIdx = 0;

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        const auto &pair = pairs[idx];
        log(QString("[密集匹配] 处理第 %1/%2 对: 图像%3 + 图像%4")
                .arg((int)idx + 1).arg((int)pairs.size())
                .arg(pair.imageId1).arg(pair.imageId2));

        // 查找该对的稀疏匹配点
        std::vector<cv::Point2d> pts1, pts2;
        for (const auto &mp : matchPairs) {
            if ((mp.imageId1 == pair.imageId1 && mp.imageId2 == pair.imageId2) ||
                (mp.imageId1 == pair.imageId2 && mp.imageId2 == pair.imageId1)) {
                pts1 = mp.points1;
                pts2 = mp.points2;
                break;
            }
        }

        auto result = processPair(pair, pts1, pts2);
        results.push_back(result);

        log(QString("[密集匹配] 第 %1/%2 对完成, 深度图=%3")
                .arg((int)idx + 1).arg((int)pairs.size())
                .arg(result.depth.empty() ? "空" : "有效"));
    }

    return results;
}

// ═══════════════════════════════════════════════════════════════
// 处理单个立体匹配对（完整流程）
//   1. 极线校正 (委托 StereoRectifier)
//   2. 匹配点坐标变换到校正后坐标系 (undistortPoints)
//   3. SGBM立体匹配 + WLS滤波
//   4. 视差转深度
// ═══════════════════════════════════════════════════════════════

StereoPairResult StereoSGBMReconstructor::processPair(
    const StereoPairInfo &pair,
    const std::vector<cv::Point2d> &matchedPoints1,
    const std::vector<cv::Point2d> &matchedPoints2)
{
    StereoPairResult result;
    result.imageId1 = pair.imageId1;
    result.imageId2 = pair.imageId2;

    // 流式加载图像
    StereoPairInfo loadedPair = pair;
    if (loadedPair.image1.empty() && !loadedPair.imagePath1.isEmpty()) {
        loadedPair.image1 = cv::imread(loadedPair.imagePath1.toStdString());
    }
    if (loadedPair.image2.empty() && !loadedPair.imagePath2.isEmpty()) {
        loadedPair.image2 = cv::imread(loadedPair.imagePath2.toStdString());
    }
    if (loadedPair.image1.empty() || loadedPair.image2.empty()) {
        log(QString("[密集匹配] 图像加载失败: %1 或 %2")
                .arg(loadedPair.imagePath1).arg(loadedPair.imagePath2));
        return result;
    }

    // ── 1. 极线校正（委托 StereoRectifier） ──
    log("[密集匹配]   步骤1: 极线校正...");
    if (!m_rectifier.rectifyPair(loadedPair, result)) {
        log("[密集匹配]   步骤1: 极线校正失败");
        return result;
    }
    log(QString("[密集匹配]   步骤1: 极线校正完成, 校正图=%1x%2")
            .arg(result.rectifiedImage1.cols).arg(result.rectifiedImage1.rows));

    // ── 2. 将匹配点变换到校正后坐标系 ──
    std::vector<cv::Point2d> rectPts1, rectPts2;
    if (!matchedPoints1.empty() && !result.P1.empty()) {
        cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

        cv::Mat pts1Mat(matchedPoints1.size(), 1, CV_64FC2);
        cv::Mat pts2Mat(matchedPoints2.size(), 1, CV_64FC2);
        for (size_t i = 0; i < matchedPoints1.size(); ++i) {
            pts1Mat.at<cv::Vec2d>(static_cast<int>(i), 0) =
                cv::Vec2d(matchedPoints1[i].x, matchedPoints1[i].y);
            pts2Mat.at<cv::Vec2d>(static_cast<int>(i), 0) =
                cv::Vec2d(matchedPoints2[i].x, matchedPoints2[i].y);
        }

        cv::Mat rectPts1Mat, rectPts2Mat;
        if (!result.R1_rect.empty() && !result.R2_rect.empty()) {
            cv::undistortPoints(pts1Mat, rectPts1Mat, pair.K1, distCoeffs,
                                result.R1_rect, result.P1);
            cv::undistortPoints(pts2Mat, rectPts2Mat, pair.K2, distCoeffs,
                                result.R2_rect, result.P2);
        } else {
            cv::undistortPoints(pts1Mat, rectPts1Mat, pair.K1, distCoeffs,
                                cv::noArray(), result.P1);
            cv::undistortPoints(pts2Mat, rectPts2Mat, pair.K2, distCoeffs,
                                cv::noArray(), result.P2);
        }

        rectPts1.resize(rectPts1Mat.rows);
        rectPts2.resize(rectPts2Mat.rows);
        // cv::undistortPoints 输出与输入同深度：输入 CV_64FC2 时输出也是
        // CV_64FC2。必须按实际深度读取——用 Vec2f 读 double 数据会把每个
        // double 的位模式拆成两个 float，坐标全部变成垃圾（历史 bug：稀疏
        // 对应缺失时本段不执行，问题被掩盖）。
        const bool out64 = rectPts1Mat.depth() == CV_64F;
        for (int i = 0; i < rectPts1Mat.rows; ++i) {
            if (out64) {
                const cv::Vec2d p1 = rectPts1Mat.at<cv::Vec2d>(i, 0);
                const cv::Vec2d p2 = rectPts2Mat.at<cv::Vec2d>(i, 0);
                rectPts1[i] = cv::Point2d(p1[0], p1[1]);
                rectPts2[i] = cv::Point2d(p2[0], p2[1]);
            } else {
                const cv::Vec2f p1 = rectPts1Mat.at<cv::Vec2f>(i, 0);
                const cv::Vec2f p2 = rectPts2Mat.at<cv::Vec2f>(i, 0);
                rectPts1[i] = cv::Point2d(p1[0], p1[1]);
                rectPts2[i] = cv::Point2d(p2[0], p2[1]);
            }
        }
    }

    // ── 3. SGBM立体匹配 + WLS滤波 ──
    log("[密集匹配]   步骤3: SGBM匹配...");
    if (!stereoMatch(result, rectPts1, rectPts2)) {
        log("[密集匹配]   步骤3: SGBM匹配失败");
        return result;
    }
    log(QString("[密集匹配]   步骤3: SGBM匹配完成, 视差图=%1x%2")
            .arg(result.disparity.cols).arg(result.disparity.rows));

    // ── 4. 视差转深度 ──
    log("[密集匹配]   步骤4: 视差转深度...");
    disparityToDepth(result);
    log(QString("[密集匹配]   步骤4: 完成, 深度图=%1x%2")
            .arg(result.depth.cols).arg(result.depth.rows));

    return result;
}

// ═══════════════════════════════════════════════════════════════
// 视差范围自动估计
// ═══════════════════════════════════════════════════════════════

void StereoSGBMReconstructor::estimateDisparityRange(
    const std::vector<cv::Point2d> &points1,
    const std::vector<cv::Point2d> &points2,
    int &minDisp, int &numDisp) const
{
    if (points1.size() != points2.size() || points1.empty()) {
        minDisp = m_config.minDisparity;
        numDisp = m_config.numDisparities;
        return;
    }

    double maxD = -1e9, minD = 1e9;
    int validCount = 0;
    int negativeCount = 0;
    for (size_t i = 0; i < points1.size(); ++i) {
        double d = points1[i].x - points2[i].x;
        if (!std::isfinite(d) || d < -5000.0 || d > 5000.0) continue;
        if (d < 0.0) ++negativeCount;
        if (d > maxD) maxD = d;
        if (d < minD) minD = d;
        validCount++;
    }

    if (validCount < 3) {
        log(QString("[视差范围] 稀疏对应不足 (%1/%2)，回退默认 min=%3 num=%4")
                .arg(validCount).arg(points1.size())
                .arg(m_config.minDisparity).arg(m_config.numDisparities));
        minDisp = m_config.minDisparity;
        numDisp = m_config.numDisparities;
        return;
    }

    minDisp = static_cast<int>(std::floor(minD)) - 1;
    int maxDisp = static_cast<int>(std::ceil(maxD)) + 1;

    if (minDisp < 0) minDisp = 0;

    numDisp = maxDisp - minDisp;
    numDisp = ((numDisp + 15) / 16) * 16;
    if (numDisp <= 0) numDisp = 16;

    int maxNumDisp = ((m_config.numDisparities * 3 + 15) / 16) * 16;
    const bool capped = numDisp > maxNumDisp;
    if (capped) {
        numDisp = maxNumDisp;
    }

    // 诊断：校正后真实视差恒为正（右相机位于 +x），负值/极值说明稀疏
    // 对应中混有错匹配；范围被上限截断时真实视差可能落在搜索范围外
    log(QString("[视差范围] 校正后对应=%1/%2, 原始视差 min=%3 max=%4 (负=%5)"
                ", 估计 num=%6%7 上限=%8")
            .arg(validCount).arg(points1.size())
            .arg(minD, 0, 'f', 1).arg(maxD, 0, 'f', 1).arg(negativeCount)
            .arg(numDisp).arg(capped ? QStringLiteral(" (被上限截断!)") : QStringLiteral())
            .arg(maxNumDisp));
}
