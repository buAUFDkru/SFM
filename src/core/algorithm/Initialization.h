#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <QObject>
#include <vector>

#include <opencv2/core.hpp>

#include "data/SfmTypes.h"

/**
 * @brief 初始化结果结构体
 *
 * 保存增量式SFM初始化阶段找到的最优图像对及其相对位姿估计结果。
 * 该结果将作为 Triangulate::triangulate() 的输入，启动增量重建流程。
 */
struct InitResult
{
    int imageId1;                          ///< 第一张图像ID（作为世界坐标系参考，位姿固定为 [I|0]）
    int imageId2;                          ///< 第二张图像ID（相对图像1的位姿由R、t描述）
    cv::Mat R;                             ///< 从相机1到相机2的旋转矩阵（3×3，SO(3)）
    cv::Mat t;                             ///< 从相机1到相机2的平移向量（3×1，单位与场景尺度一致）
    cv::Mat F;                             ///< 基础矩阵（3×3），由RANSAC估计
    cv::Mat H;                             ///< 单应矩阵（3×3），由RANSAC估计
    int modelChoice;                       ///< GRIC模型选择结果：0=基础矩阵F，1=单应矩阵H
    std::vector<cv::Point2d> points1;     ///< 图像1中的内点像素坐标
    std::vector<cv::Point2d> points2;     ///< 图像2中的内点像素坐标
    std::vector<cv::DMatch> inlierMatches; ///< 内点匹配索引列表
    double score;                          ///< 综合评分（内点数 × 角度权重 × 模型加分）
    double baseline;                       ///< 基线长度 ‖t‖₂（用于评估重建质量）
    double medianAngle;                    ///< 中位三角化角度（度），优先区间约16°-30°（COLMAP）
};

/**
 * @brief 初始化类 —— 增量式SFM的初始化阶段
 *
 * 无跨调用可变状态，可栈上构造（ADR-003 DI）。
 *
 * 负责在增量式SFM重建开始前完成以下关键步骤：
 *
 * 1. **寻找最优初始化图像对**（findInitialPair）
 *    - 遍历所有匹配对，对每个候选对估计相对位姿
 *    - 三角化内点并计算中位三角化角度
 *    - 采用COLMAP式基线评分策略综合打分，选择最优对
 *
 * 2. **2D-2D相对定向**（selectModel）
 *    - 同时用RANSAC估计基础矩阵F和单应矩阵H
 *    - 用GRIC（几何鲁棒信息准则）选择更优模型
 *    - F模型适合一般3D场景，H模型适合平面占主导的场景
 *
 * 3. **相对位姿恢复**
 *    - recoverPoseFromF：从F矩阵通过本质矩阵E分解得到R、t
 *    - recoverPoseFromH：从H矩阵分解得到R、t，通过正深度验证选最优解
 *
 * ### 基线评分原理（参考 COLMAP init_min_tri_angle=16°）
 *
 * 好的初始化基线需要在深度精度和匹配可靠性之间取得平衡：
 * - 基线太短 → 三角化角度小 → 深度估计不确定性大（无人机相邻帧常见 ~6°）
 * - 基线太长 → 视角差异大 → 特征匹配困难、外点多
 * - COLMAP 对种子对硬性要求中位三角化角 > 16°；本实现以 16°–30° 为权重峰值，
 *   并在存在 ≥12° 候选时屏蔽弱视差对，避免高内点相邻帧压过宽基线对
 *
 * 角度权重函数：
 * ```
 *   angle < 5°        → 权重=0（退化）
 *   5° ≤ angle < 12°  → 弱权重（仅兜底）
 *   12° ≤ angle < 16° → 爬升到 1
 *   16° ≤ angle ≤ 30° → 权重=1（优先区间）
 *   30° < angle ≤ 60° → 缓降到 0.5
 *   angle > 60°       → 继续缓降（保留大视角区分度）
 * ```
 */
class Initialization : public QObject
{
    Q_OBJECT

public:
    explicit Initialization(QObject *parent = nullptr);
    ~Initialization() override = default;

    /**
     * @brief 寻找最优初始化图像对
     *
     * 对所有匹配对执行以下评估流程：
     * 1. 跳过匹配点数 < 30 的候选对
     * 2. 根据GRIC选择的模型（F或H）恢复相对位姿R、t
     * 3. 对内点进行三角化，计算每个点的三角化角度
     * 4. 取中位角度，通过角度权重函数计算权重
     * 5. 综合评分 = 有效内点数 × 角度权重 × 模型加分（F模型×1.2）
     * 6. 选择评分最高的候选对作为初始化对
     *
     * @param features 图像特征点列表（每张图的关键点和描述符）
     * @param matchPairs 匹配对列表（已通过GRIC验证，包含F/H矩阵）
     * @param K 相机内参矩阵（3×3）
     * @return 最优初始化结果（score<0表示未找到合适对）
     */
    InitResult findInitialPair(const std::vector<FeatureResult> &features,
                               const std::vector<MatchPair> &matchPairs,
                               const cv::Mat &K);

    /**
     * @brief 寻找排序后的初始化候选对（支持多候选重试）
     *
     * 与 findInitialPair 评估逻辑相同，但返回按评分降序排列的 Top-K 候选，
     * 供上层在某个种子对重建失败时换用下一个候选（对应 COLMAP
     * init_max_reg_trials 的多轮初始化尝试）。
     *
     * @param features 图像特征点列表
     * @param matchPairs 匹配对列表
     * @param K 相机内参矩阵
     * @param topK 返回的候选数量上限；<=0 表示返回全部
     * @return 按评分降序排列的候选列表（为空表示未找到任何可用对）
     */
    std::vector<InitResult> findInitialPairCandidates(
        const std::vector<FeatureResult> &features,
        const std::vector<MatchPair> &matchPairs,
        const cv::Mat &K,
        int topK = 1);

    /**
     * @brief 2D-2D相对定向：估计F和H，用GRIC准则选择最优模型
     *
     * 同时使用RANSAC估计基础矩阵F和单应矩阵H，计算两种模型的GRIC得分，
     * 选择得分更低（更优）的模型。
     *
     * GRIC（Geometric Robust Information Criterion）公式：
     *   GRIC = Σ ρ(eᵢ²/σ²) + λ₁·d·n + λ₂·k
     * 其中：
     *   ρ(e) = min(e, λ₃(r-d))  — 鲁棒截断函数
     *   d — 模型自由度（F:3, H:2）
     *   k — 模型参数个数（F:7, H:8）
     *   r — 数据维度（2D点=4）
     *   λ₁, λ₂, λ₃ — 惩罚系数
     *
     * F模型的GRIC通常低于H模型（除非场景近似平面），
     * 因此F模型被优先选择，避免平面退化导致的重建失败。
     *
     * @param pts1 图像1匹配点像素坐标
     * @param pts2 图像2匹配点像素坐标
     * @param F 输出：估计的基础矩阵（3×3）
     * @param H 输出：估计的单应矩阵（3×3）
     * @param outInlierMask 输出：选中模型的内点掩码（与输入点一一对应）
     * @param sigma 噪声标准差（像素，默认1.0）
     * @param ransacThreshold RANSAC重投影阈值（像素，默认3.0）
     * @return 0=选择F模型，1=选择H模型
     */
    int selectModel(const std::vector<cv::Point2d> &pts1,
                    const std::vector<cv::Point2d> &pts2,
                    cv::Mat &F, cv::Mat &H,
                    std::vector<uchar> &outInlierMask,
                    double sigma = 1.0,
                    double ransacThreshold = 3.0);

    /**
     * @brief 从基础矩阵F恢复相对位姿R、t
     *
     * 流程：
     * 1. 由F计算本质矩阵E = KᵀFK
     * 2. 调用cv::recoverPose分解E，得到4组候选解
     * 3. 通过cheirality约束（正深度验证）选择唯一解
     *
     * @param pts1 图像1匹配点像素坐标
     * @param pts2 图像2匹配点像素坐标
     * @param K 相机内参矩阵（3×3）
     * @param F 基础矩阵（3×3）
     * @param R 输出：旋转矩阵（3×3）
     * @param t 输出：平移向量（3×1，归一化）
     * @param inlierMask 输出：recoverPose的内点掩码
     * @return 是否成功恢复位姿（内点数≥8视为成功）
     */
    bool recoverPoseFromF(const std::vector<cv::Point2d> &pts1,
                          const std::vector<cv::Point2d> &pts2,
                          const cv::Mat &K, const cv::Mat &F,
                          cv::Mat &R, cv::Mat &t,
                          std::vector<uchar> &inlierMask);

    /**
     * @brief 从单应矩阵H恢复相对位姿R、t
     *
     * 流程：
     * 1. 调用cv::decomposeHomographyMat分解H，最多得到4组候选解(R,t,n)
     * 2. 对每组解，取前20个匹配点三角化
     * 3. 统计在两个相机前方（正深度）的点数
     * 4. 选择正深度点数最多的解
     *
     * 注意：H分解的平移t只有方向（单位向量），尺度不确定。
     * 实际尺度由后续三角化点的深度决定。
     *
     * @param pts1 图像1匹配点像素坐标
     * @param pts2 图像2匹配点像素坐标
     * @param K 相机内参矩阵（3×3）
     * @param H 单应矩阵（3×3）
     * @param R 输出：旋转矩阵（3×3）
     * @param t 输出：平移向量（3×1）
     * @return 是否成功恢复位姿
     */
    bool recoverPoseFromH(const std::vector<cv::Point2d> &pts1,
                          const std::vector<cv::Point2d> &pts2,
                          const cv::Mat &K, const cv::Mat &H,
                          cv::Mat &R, cv::Mat &t);

private:
    Initialization(const Initialization &) = delete;
    Initialization &operator=(const Initialization &) = delete;
};

#endif // INITIALIZATION_H