#ifndef TWOVIEWGEOMETRY_H
#define TWOVIEWGEOMETRY_H

#include <vector>

#include <opencv2/core.hpp>

#include "data/SfmTypes.h"

/**
 * @brief 两视图几何：已知内参时直接估 E，并判定场景配置；附极线引导匹配
 *
 * 对应 COLMAP 的 `estimators/two_view_geometry.h`：
 *
 * | config | 含义 | 下游如何使用 |
 * |---|---|---|
 * | CALIBRATED   | 内参已知，本质矩阵 E 有效（普通三维场景） | 用 E 恢复相对位姿，匹配可信 |
 * | UNCALIBRATED | 内参未知，只能用基础矩阵 F | 仍可建边，位姿精度略差 |
 * | PLANAR       | 单应 H 解释的内点明显多于 E/F（近似平面场景） | 可建边，但不宜做种子对 |
 * | PANORAMIC    | E 有效但 cheirality 通过率过低（近似纯旋转） | 不可恢复平移，慎选 |
 * | DEGENERATE   | 内点不足 8 或模型估计失败 | 直接丢弃该对 |
 *
 * 关键改进点（相对旧实现）：
 * - 旧实现一律走「F + H + GRIC」，即使传入了已知 K 也要先估 F 再做 `E = KᵀFK`，
 *   一次多余的两次矩阵变换会放大误差；这里在 K 有效时直接估 E。
 * - RANSAC 阈值旧实现固定 3 px，不随分辨率与焦距缩放；这里归一化到
 *   「单位焦距」空间，阈值语义与图像分辨率解耦。
 * - 新增 `guidedMatching()`：用已估出的 E/H 做极线（或单应）约束，
 *   在被 Lowe 比率拒掉的候选里二次捞取真匹配，提升宽基线召回。
 */

/**
 * @brief 自适应 RANSAC 阈值（像素）
 *
 * 阈值固定 3 px 对 12MP 航拍图偏严、对 640×480 偏松。这里按图像
 * 长边缩放并夹在 [1, 4] px：
 *   - 长边 4000 → 4.0 px（与 COLMAP 默认量级一致）
 *   - 长边 1000 → 1.0 px
 * 调用方仍可用 `ransacThreshold` 显式覆盖。
 */
double adaptiveRansacThresholdPx(int imageWidth, int imageHeight);

/// 两视图几何估计参数
struct TwoViewGeometryOptions
{
    int    minNumInliers   = 15;      ///< 判定模型有效的最少内点数
    double ransacMaxErrorPx = 3.0;    ///< RANSAC 阈值（像素，已按分辨率自适应）
    /// RANSAC 置信度。取 0.999 而非 0.9999：内点率低时所需迭代数按
    /// log(1-conf)/log(1-p^n) 暴涨，0.9999 会让弱对把单次几何估计拖到秒级。
    double confidence       = 0.999;
    /// RANSAC 迭代上限（COLMAP 取 1000）。同理：低内点率下会被打满，
    /// 上限直接决定弱对的最坏耗时，这里留 1000 与 COLMAP 对齐。
    int    maxNumTrials     = 500;
    /// 本质矩阵 RANSAC 的抽样上限（点数）。5 点法每次迭代要全量打分，
    /// 直接在两三千个匹配上跑会慢到几十分钟；抽样后内点率不变、迭代次数不变，
    /// 但每次迭代的打分成本降一个量级。最终内点数仍按全量解析统计。
    int    essentialRansacMaxPoints = 600;
    /// H / F 的 RANSAC 抽样上限（点数），机制同上（模型来自抽样，统计在全量）
    int    ransacSampleCap = 800;
    bool   forceUncalibrated = false; ///< 调试用：即使有 K 也强制走 F 路径
    /// 单应内点数需超过该倍数的 E/F 内点数才判定为 PLANAR（防抖动误判）
    double planarDominanceRatio = 1.1;
};

/// 两视图几何估计结果
struct TwoViewGeometry
{
    cv::Mat E;                        ///< 本质矩阵（归一化坐标：x₂ₙᵀ·E·x₁ₙ = 0）
    cv::Mat F;                        ///< 基础矩阵（**像素坐标**；CALIBRATED 下由 K⁻ᵀEK⁻¹ 导出）
    cv::Mat H;                        ///< 单应矩阵（像素坐标）
    cv::Mat R;                        ///< 由 E 恢复的旋转（仅 CALIBRATED 有效，cam1→cam2）
    cv::Mat t;                        ///< 由 E 恢复的平移（单位向量，仅 CALIBRATED 有效）
    int config = TWO_VIEW_UNDEFINED;  ///< TwoViewConfig
    int numInliers = 0;               ///< 选中模型的内点数
    double inlierRatio = 0.0;         ///< 内点率
    // ── 诊断量（调阈值时看这几个数即可判断是哪一类场景）──
    int    numEInliers = 0;           ///< 本质矩阵内点数（0 = 未走 E 路径）
    int    numHInliers = 0;           ///< 单应内点数
    double poseFrontRatio = 0.0;      ///< E 分解位姿的 cheirality 通过率（1=姿态无歧义）
    std::vector<uchar> inlierMask;    ///< 内点掩码（与输入点一一对应）

    bool valid() const { return config != TWO_VIEW_DEGENERATE &&
                                config != TWO_VIEW_UNDEFINED && numInliers >= 8; }
    /// 是否可用于恢复相对位姿（PANORAMIC 为纯旋转，无法恢复平移）
    bool poseValid() const { return config == TWO_VIEW_CALIBRATED && !R.empty() && !t.empty(); }
};

/**
 * @brief 估计两视图几何
 *
 * @param pts1/pts2 一对图像的匹配点（像素坐标，数量相同）
 * @param K 相机内参；为空或非法时自动退化为 F 路径
 * @param opt 估计参数
 * @return 几何结果；`valid()` 为 false 表示该对应丢弃
 */
TwoViewGeometry estimateTwoViewGeometry(const std::vector<cv::Point2d> &pts1,
                                        const std::vector<cv::Point2d> &pts2,
                                        const cv::Mat &K,
                                        const TwoViewGeometryOptions &opt);

/**
 * @brief 由已知 E 与内参恢复相对位姿（封装 cv::recoverPose 的归一化约定）
 *
 * @param E 归一化坐标下的本质矩阵
 * @param pts1/pts2 像素坐标匹配点
 * @param K 内参
 * @param R/t 输出相对位姿（cam1→cam2）
 * @param mask 输出内点掩码
 * @return 成功返回内点数，失败返回 0
 */
int recoverPoseFromE(const cv::Mat &E,
                     const std::vector<cv::Point2d> &pts1,
                     const std::vector<cv::Point2d> &pts2,
                     const cv::Mat &K,
                     cv::Mat &R, cv::Mat &t,
                     std::vector<uchar> &mask);

/**
 * @brief 极线 / 单应引导的二次匹配（guided matching）
 *
 *  Lowe 比率测试在宽基线下会误杀大量真匹配（视角变化使描述符漂移）。
 *  这里利用已估出的模型做空间约束：对每个尚未匹配的查询点，在其
 *  KNN 候选（k ≥ 2）中只保留几何误差 ≤ maxErrorPx 的，再在这些
 *  「幸存者」内部重做比率测试。
 *
 *  约束叠加后假阳性概率很低：极线带宽通常只占图像宽度的千分之一量级。
 *
 * @param knn 查询点 → 候选匹配（每个查询点 k 个候选，按描述符距离升序）
 * @param pts1/pts2 像素坐标关键点
 * @param geom 已估出的两视图几何（提供 F 或 H）
 * @param maxErrorPx 几何误差阈值（像素）
 * @param ratioThreshold 幸存者内部的比率阈值
 * @param usedQuery 已占用查询点掩码（为空表示全部可用）
 * @param usedTrain 已占用目标点掩码（为空表示全部可用）
 * @param outMatches 追加输出的新匹配
 * @return 新增匹配数
 */
int guidedMatching(const std::vector<std::vector<cv::DMatch>> &knn,
                   const std::vector<cv::Point2d> &pts1,
                   const std::vector<cv::Point2d> &pts2,
                   const TwoViewGeometry &geom,
                   double maxErrorPx,
                   double ratioThreshold,
                   const std::vector<uchar> &usedQuery,
                   const std::vector<uchar> &usedTrain,
                   std::vector<cv::DMatch> &outMatches);

#endif // TWOVIEWGEOMETRY_H
