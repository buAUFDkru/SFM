#include "StereoRectifier.h"

#include <algorithm>
#include <cmath>
#include <QDebug>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// ═══════════════════════════════════════════════════════════════
// 日志
// ═══════════════════════════════════════════════════════════════

void StereoRectifier::log(const QString &message) const
{
    if (m_logCallback) {
        m_logCallback(message);
    } else {
        qDebug().noquote() << message;
    }
}

// ═══════════════════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════════════════

double StereoRectifier::computeBaseline(const cv::Mat &R1, const cv::Mat &t1,
                                        const cv::Mat &R2, const cv::Mat &t2)
{
    cv::Mat C1 = R1.t() * (-t1);
    cv::Mat C2 = R2.t() * (-t2);
    return cv::norm(C1 - C2);
}

double StereoRectifier::computeViewAngle(const cv::Mat &R1, const cv::Mat &R2)
{
    // R^T * [0;0;1] = R的第3列
    cv::Mat d1 = R1.col(2);
    cv::Mat d2 = R2.col(2);
    double cosAngle = d1.dot(d2);
    cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
    return std::acos(cosAngle) * 180.0 / CV_PI;
}

// ═══════════════════════════════════════════════════════════════
// 极线校正
//
// 使用自定义Bouguet算法变体进行极线校正，使两幅图像的
// 极线水平对齐。校正后同一物体在左右图中的y坐标相同，
// 只需沿x方向搜索匹配，大幅降低匹配搜索空间。
//
// 关键步骤：
//   1. 计算相机1到相机2的相对位姿
//   2. 构造 R_rect 将基线 T 旋转到 x 轴负方向
//   3. 计算 R1_rect, R2_rect 和统一新内参
//   4. 通过角点投影确定新图像尺寸和主点
//   5. 构造 P1, P2, Q 矩阵
//   6. initUndistortRectifyMap + remap 重映射图像
// ═══════════════════════════════════════════════════════════════

bool StereoRectifier::rectifyPair(const StereoPairInfo &pair, StereoPairResult &result)
{
    result.imageId1 = pair.imageId1;
    result.imageId2 = pair.imageId2;

    if (pair.image1.empty() || pair.image2.empty()) {
        return false;
    }

    cv::Size imgSize = pair.image1.size();

    // ── 计算相对位姿 ──
    // 位姿约定: P_cam = R * P_world + t (world→camera)
    // 相机1坐标系到相机2坐标系的变换:
    //   P_cam1 = R1 * P_world + t1  →  P_world = R1^T * (P_cam1 - t1)
    //   P_cam2 = R2 * P_world + t2
    //          = R2 * R1^T * P_cam1 + (t2 - R2 * R1^T * t1)
    // 所以: R = R2 * R1^T,  T = t2 - R2 * R1^T * t1
    cv::Mat R = pair.R2 * pair.R1.t();
    cv::Mat T = pair.t2 - pair.R2 * pair.R1.t() * pair.t1;

    // ── 自定义极线校正（替代 cv::stereoRectify） ──
    // cv::stereoRectify 的 Bouguet 算法在基线不沿x轴时会产生错误结果
    // （P2的tx分量接近0，导致深度映射错误）。
    // 参考Python项目 StereoRectify-main 的实现，自行构造R1/R2和新内参。

    // Step 1: 构造 R_rect — 将T旋转到x轴负方向 (-1,0,0)
    // 这样校正后视差 d = x1 - x2 = -fx*T_rect.x/Z > 0（T_rect.x = -baseline），
    // 极线水平对齐，SGBM 在正向范围搜索可命中
    double Tx = T.at<double>(0), Ty = T.at<double>(1), Tz = T.at<double>(2);
    double Tnorm = std::sqrt(Tx*Tx + Ty*Ty + Tz*Tz);
    double baseline = Tnorm;

    cv::Mat R_rect = cv::Mat::eye(3, 3, CV_64F);
    if (Tnorm > 1e-10) {
        // 目标方向: x轴负方向 (-1, 0, 0)
        // 旋转轴 = T_unit × target
        cv::Mat Tunit = (cv::Mat_<double>(3,1) << Tx/Tnorm, Ty/Tnorm, Tz/Tnorm);
        cv::Mat target = (cv::Mat_<double>(3,1) << -1.0, 0.0, 0.0);

        cv::Mat rotAxisVec = Tunit.cross(target);
        double sinAngle = cv::norm(rotAxisVec);
        double cosAngle = Tunit.dot(target);

        if (sinAngle > 1e-8) {
            // 使用 cv::Rodrigues 计算旋转矩阵
            cv::Mat k = rotAxisVec / sinAngle;
            double angle = std::acos(std::max(-1.0, std::min(1.0, cosAngle)));
            cv::Mat rotVec = k * angle;
            cv::Rodrigues(rotVec, R_rect);
        } else if (cosAngle < 0) {
            // T与x轴正方向平行（方向相反），需旋转180度
            // 选一个与T正交的轴
            cv::Mat u = (cv::Mat_<double>(3,1) << 0.0, 0.0, 1.0);
            if (std::abs(Tunit.dot(u)) > 0.9) {
                u = (cv::Mat_<double>(3,1) << 0.0, 1.0, 0.0);
            }
            // Householder反射: R = I - 2*u*u^T (绕u旋转180度)
            R_rect = cv::Mat::eye(3, 3, CV_64F) - 2.0 * u * u.t();
        }
        // cosAngle ≈ 1: T已沿x轴负方向，无需旋转
    }

    // Step 2: 计算 R1_rect, R2_rect
    // R2 = R_rect (将右相机坐标系旋转，使基线沿x轴)
    // R1 = R_rect * R (左相机也做相应旋转，使两相机相对旋转为单位阵)
    cv::Mat R1_rect = R_rect * R;   // 左相机校正旋转
    cv::Mat R2_rect = R_rect;       // 右相机校正旋转

    // Step 3: 计算新的统一内参矩阵
    // 取两相机焦距的平均值，确保校正后两图投影尺度一致
    double fx1 = pair.K1.at<double>(0, 0);
    double fy1 = pair.K1.at<double>(1, 1);
    double fx2 = pair.K2.at<double>(0, 0);
    double fy2 = pair.K2.at<double>(1, 1);
    double fx_avg = (fx1 + fx2) / 2.0;
    double fy_avg = (fy1 + fy2) / 2.0;
    double fx_new = (fx_avg + fy_avg) / 2.0;  // 统一焦距
    double fy_new = fx_new;

    // Step 4: 计算角点投影，确定新图像尺寸和主点
    // 将原图四角通过 R1/R2 旋转后投影到新内参下，计算包围盒
    auto projectCorners = [](const cv::Mat &K, const cv::Mat &R_rect_mat,
                             const cv::Mat &K_new, int w, int h) -> std::vector<cv::Point2d> {
        cv::Mat Kinv = K.inv();
        std::vector<cv::Point2d> corners;
        double pts[4][2] = {{0, 0}, {(double)w, 0}, {0, (double)h}, {(double)w, (double)h}};
        for (int i = 0; i < 4; ++i) {
            cv::Mat p = (cv::Mat_<double>(3,1) << pts[i][0], pts[i][1], 1.0);
            cv::Mat camCoord = R_rect_mat * (Kinv * p);  // 旋转后的归一化坐标
            cv::Mat proj = K_new * camCoord;
            corners.push_back(cv::Point2d(proj.at<double>(0)/proj.at<double>(2),
                                           proj.at<double>(1)/proj.at<double>(2)));
        }
        return corners;
    };

    // 先用临时内参（主点在图像中心）计算角点投影
    cv::Mat K_new_tmp = (cv::Mat_<double>(3,3) <<
        fx_new, 0, imgSize.width/2.0,
        0, fy_new, imgSize.height/2.0,
        0, 0, 1);

    auto corners1 = projectCorners(pair.K1, R1_rect, K_new_tmp, imgSize.width, imgSize.height);
    auto corners2 = projectCorners(pair.K2, R2_rect, K_new_tmp, imgSize.width, imgSize.height);

    // 计算包围盒
    double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
    for (const auto &c : corners1) {
        x_min = std::min(x_min, c.x); x_max = std::max(x_max, c.x);
        y_min = std::min(y_min, c.y); y_max = std::max(y_max, c.y);
    }
    for (const auto &c : corners2) {
        x_min = std::min(x_min, c.x); x_max = std::max(x_max, c.x);
        y_min = std::min(y_min, c.y); y_max = std::max(y_max, c.y);
    }

    // 计算新主点（使图像居中）
    double cx_new = (x_max + x_min) / 2.0;
    double cy_new = (y_max + y_min) / 2.0;

    // 计算新图像尺寸（保留完整视场）
    int newWidth = static_cast<int>(std::ceil(x_max - x_min));
    int newHeight = static_cast<int>(std::ceil(y_max - y_min));
    // 确保尺寸合理
    newWidth = std::max(newWidth, imgSize.width / 2);
    newHeight = std::max(newHeight, imgSize.height / 2);
    newWidth = std::min(newWidth, imgSize.width * 3);
    newHeight = std::min(newHeight, imgSize.height * 3);

    // 构造最终的新内参矩阵（左右相同）
    cv::Mat K_new = (cv::Mat_<double>(3,3) <<
        fx_new, 0, cx_new,
        0, fy_new, cy_new,
        0, 0, 1);

    // Step 5: 构造投影矩阵 P1, P2
    // P1 = K_new * [I | 0]  (左相机在校正后坐标系原点)
    // P2 = K_new * [I | T_rect]  (右相机中心位于 -T_rect = (+baseline,0,0))
    // T_rect = R_rect * T = [-baseline, 0, 0]^T（Step1 将 T 旋转到 -x 方向）
    // 由 P_c2' = P_c1' + T_rect 知右相机中心在左校正坐标系（共享坐标）中为 +x：
    //   x1 = fx*X/Z + cx, x2 = fx*(X - baseline)/Z + cx
    //   d = x1 - x2 = fx*baseline/Z > 0（与 SGBM 视差定义一致）
    cv::Mat T_rect = R_rect * T;  // 应该接近 [-baseline, 0, 0]^T

    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    K_new.copyTo(P1(cv::Rect(0, 0, 3, 3)));

    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    K_new.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    P2.at<double>(0, 3) = K_new.at<double>(0, 0) * T_rect.at<double>(0);  // fx * Tx = -fx*baseline
    P2.at<double>(1, 3) = K_new.at<double>(1, 1) * T_rect.at<double>(1);  // fy * Ty
    P2.at<double>(2, 3) = T_rect.at<double>(2);                            // Tz

    // Step 6: 构造 Q 矩阵（视差到3D重投影，与 OpenCV stereoRectify 标准形式一致）
    // Q = [1  0    0     -cx_new   ]
    //     [0  1    0     -cy_new   ]
    //     [0  0    0      fx_new   ]
    //     [0  0  -1/Tx   (cx1-cx2)/Tx ]
    // 其中 Tx = P2[0][3]/fx = -baseline，故 Q[3][2] = -1/Tx = +1/baseline
    double Tx_rect = T_rect.at<double>(0);  // 应为 -baseline
    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    Q.at<double>(0, 0) = 1.0;
    Q.at<double>(0, 3) = -cx_new;
    Q.at<double>(1, 1) = 1.0;
    Q.at<double>(1, 3) = -cy_new;
    Q.at<double>(2, 3) = fx_new;
    Q.at<double>(3, 2) = -1.0 / Tx_rect;  // 1/baseline (Tx_rect = -baseline)
    Q.at<double>(3, 3) = 0.0;  // cx1 == cx2 (统一内参)

    // 畸变系数设为零（假设图像已校正）
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

    // ── 计算校正映射表 ──
    cv::Size newSize(newWidth, newHeight);
    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(pair.K1, distCoeffs, R1_rect, K_new,
                                newSize, CV_16SC2, map1x, map1y);
    cv::initUndistortRectifyMap(pair.K2, distCoeffs, R2_rect, K_new,
                                newSize, CV_16SC2, map2x, map2y);

    // ── 重映射：将原图像素映射到校正后的图像 ──
    cv::remap(pair.image1, result.rectifiedImage1, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(pair.image2, result.rectifiedImage2, map2x, map2y, cv::INTER_LINEAR);

    // ── 保存校正结果 ──
    result.Q = Q;                       // 重投影矩阵: [x y d 1] -> [X Y Z 1]
    result.P1 = P1;                     // 左相机校正后投影矩阵
    result.P2 = P2;                     // 右相机校正后投影矩阵
    result.K_rect = K_new.clone();      // 校正后内参
    result.R1_rect = R1_rect.clone();   // 左图校正旋转矩阵
    result.R2_rect = R2_rect.clone();   // 右图校正旋转矩阵
    result.baseline = baseline;

    return true;
}
