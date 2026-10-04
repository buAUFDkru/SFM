#ifndef SFMTYPES_H
#define SFMTYPES_H

#include <QImage>
#include <QString>

#include <array>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

#include "plugin/PluginTypes.h"

/**
 * @brief 共享工具函数
 *
 * 本文件定义 SFM 重建流程中使用的数据结构（算法层）。
 * 跨 DLL 的传输结构（ImageEntry/FeatureResult）位于 plugin/PluginTypes.h。
 * 依赖 PCL 的 ReconstructionOutput 保留到重建管线落地阶段（P3b）再引入。
 */

/**
 * @brief 根据图像尺寸估计相机内参矩阵
 *
 * 使用针孔相机模型，根据图像尺寸估算一个默认的内参矩阵。
 * 假设水平视野约为60度（Pi/3），垂直视野由图像宽高比决定。
 *
 * 计算公式：
 * - fy = height / 2 / (Pi/6) = 3 * height / Pi
 * - fx = fy (假设像素是正方形)
 * - cx = width / 2
 * - cy = height / 2
 *
 * @param width 图像宽度（像素）
 * @param height 图像高度（像素）
 * @return 3x3内参矩阵K
 */
inline cv::Mat estimateK(int width, int height)
{
    const double kPi = 3.141592653589793;
    double fy = static_cast<double>(height) * 0.5 / (kPi / 6.0);
    double fx = fy;
    double cx = static_cast<double>(width)  * 0.5;
    double cy = static_cast<double>(height) * 0.5;
    return (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
}

inline cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K)
{
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

/// 图像序列类型别名（按顺序存储所有输入图像）
using ImageSequence = std::vector<ImageEntry>;

/**
 * @brief 两视图几何配置（对齐 COLMAP TwoViewGeometry::Configuration）
 *
 * 由 `TwoViewGeometry` 估计产出，记录该图像对所属的场景类型。
 * 下游（初始化选种子对、建边决策）据此判断这对匹配能用到什么程度。
 */
enum TwoViewConfig
{
    TWO_VIEW_UNDEFINED    = 0,  ///< 未估计
    TWO_VIEW_DEGENERATE   = 1,  ///< 内点不足 / 模型估计失败 → 该对应丢弃
    TWO_VIEW_CALIBRATED   = 2,  ///< 内参已知，本质矩阵 E 有效 → 可恢复相对位姿
    TWO_VIEW_UNCALIBRATED = 3,  ///< 内参未知，仅基础矩阵 F 有效
    TWO_VIEW_PLANAR       = 4,  ///< 单应主导（近似平面场景）→ 不宜作种子对
    TWO_VIEW_PANORAMIC    = 5   ///< 近似纯旋转 → 平移不可恢复，不宜作种子对
};

/**
 * @brief 图像对匹配结果（算法层）
 *
 * 存储一对图像之间的特征匹配结果，包括匹配点坐标和统计信息。
 * 由 core 匹配编排（MatcherService）组装，区别于插件传输层 MatchPairResult。
 */
struct MatchPair
{
    int imageId1;                           ///< 图像A的ID
    int imageId2;                           ///< 图像B的ID
    std::vector<cv::DMatch> matches;       ///< 过滤后的匹配对（query与train索引）
    std::vector<cv::Point2d> points1;      ///< 图像A匹配点的像素坐标
    std::vector<cv::Point2d> points2;      ///< 图像B匹配点的像素坐标
    int matchCount = 0;                     ///< 有效匹配数量
    double avgDistance = 0.0;              ///< 平均描述符距离（衡量匹配质量）
    int modelChoice = 0;                    ///< 选中模型：0=F/E（对极），1=H（单应）
    cv::Mat F;                              ///< 基础矩阵（3×3，像素坐标）
    cv::Mat H;                              ///< 单应矩阵（3×3，像素坐标）
    cv::Mat E;                              ///< 本质矩阵（3×3，**归一化坐标**：x₂ₙᵀ·E·x₁ₙ=0）
    int geometryConfig = TWO_VIEW_UNDEFINED;///< 两视图几何配置（TwoViewConfig）
};

/**
 * @brief 光束法平差观测数据
 *
 * 记录每个三维点在某张图像上的二维观测，用于BA优化。
 */
struct BAObservation
{
    int pointId;                   ///< 三维点的全局索引
    int imageId;                   ///< 观测该点的图像ID
    double obsX;                   ///< 观测到的像素x坐标
    double obsY;                   ///< 观测到的像素y坐标
};

/**
 * @brief 关键点到三维点的映射类型
 *
 * 第一层key为imageId，第二层key为keypointIdx，value为对应的3D点索引。
 * 用于快速查找某个关键点是否已三角化，以及对应的三维点ID。
 *
 * 示例：kpTo3D[0][15] = 123 表示图像0的关键点15已对应到三维点123
 */
using KpTo3DMap = std::map<int, std::map<int, int>>;

/**
 * @brief 三维空间点结构
 *
 * 表示重建后的一个三维点，包含位置、颜色和观测信息。
 */
struct Point3D
{
    int id;                                ///< 点的全局唯一ID
    cv::Point3d position;                   ///< 三维坐标（米）
    cv::Vec3b color;                       ///< RGB颜色值（0-255）
    std::vector<std::pair<int, int>> observations; ///< 观测列表，每个元素为(imageId, keypointIdx)
};

/**
 * @brief 相机位姿结构
 *
 * 表示一个相机在三维空间中的位置和朝向。
 * 使用相对于某个世界坐标系的旋转矩阵R和平移向量t描述。
 * 从相机坐标系到世界坐标系的变换：P_world = R^T * (P_cam - t)
 */
struct CameraPose
{
    int imageId;                           ///< 对应图像的ID
    int imageWidth = 0;                    ///< 原始图像宽度
    int imageHeight = 0;                   ///< 原始图像高度
    cv::Mat R;                             ///< 旋转矩阵（3×3，world→camera: P_cam = R*P_world + t）
    cv::Mat t;                             ///< 平移向量（3×1，world→camera: P_cam = R*P_world + t）
    cv::Mat K;                             ///< 内参矩阵（3×3）
    cv::Mat distortion;                    ///< 畸变系数（5×1，k1,k2,p1,p2,k3）
    bool registered = false;               ///< 是否已通过PnP成功注册到重建中
};

/**
 * @brief 无向图 —— 匹配图边结构
 *
 * 表示一对图像之间的匹配关系，边的权重为过滤后的匹配点数量。
 */
struct ViewGraphEdge
{
    int nodeA;       ///< 图像A的ID（节点A）
    int nodeB;       ///< 图像B的ID（节点B）
    int matchCount;  ///< 匹配点数量（边权重）
};

/**
 * @brief 无向图 —— 匹配图数据结构
 *
 * 图像为节点，过滤后的匹配对为边。用于分析视图连接关系、
 * 选择最优种子对、识别连通分量等。
 *
 * 邻接表 adj[i] 存储与节点i相连的所有边在 edges 中的索引。
 */
struct ViewGraph
{
    int numNodes;                            ///< 节点总数（=图像数量）
    std::vector<ViewGraphEdge> edges;       ///< 边列表
    std::vector<std::vector<int>> adj;       ///< 邻接表：adj[i] = {连接到i的边在edges中的索引}

    /**
     * @brief 从匹配对列表构建匹配图
     *
     * 遍历所有匹配对，为每对图像创建一条无向边，同时构建邻接表。
     *
     * @param matchPairs 过滤后的匹配对列表
     * @param numImages 图像总数
     * @return 构建好的匹配图
     */
    static ViewGraph buildFromPairs(const std::vector<MatchPair> &matchPairs,
                                     int numImages)
    {
        ViewGraph graph;
        graph.numNodes = numImages;
        graph.adj.resize(numImages);

        graph.edges.reserve(matchPairs.size());
        for (const auto &mp : matchPairs) {
            int edgeIdx = static_cast<int>(graph.edges.size());
            graph.edges.push_back({mp.imageId1, mp.imageId2, mp.matchCount});
            graph.adj[mp.imageId1].push_back(edgeIdx);
            graph.adj[mp.imageId2].push_back(edgeIdx);
        }

        return graph;
    }

    /**
     * @brief 获取节点i的度（有多少个邻居）
     * @param i 节点索引
     * @return 邻居数量
     */
    int degree(int i) const
    {
        return static_cast<int>(adj[i].size());
    }

    /**
     * @brief 获取节点i的所有邻居节点ID
     * @param i 节点索引
     * @return 邻居节点ID列表
     */
    std::vector<int> neighbors(int i) const
    {
        std::vector<int> result;
        result.reserve(adj[i].size());
        for (int edgeIdx : adj[i]) {
            const auto &e = edges[edgeIdx];
            int nb = (e.nodeA == i) ? e.nodeB : e.nodeA;
            result.push_back(nb);
        }
        return result;
    }

    /**
     * @brief 获取连接节点i和j的边权重（匹配点数），若不相连则返回0
     * @param i 节点A
     * @param j 节点B
     * @return 匹配点数，不连通返回0
     */
    int edgeWeight(int i, int j) const
    {
        for (int edgeIdx : adj[i]) {
            const auto &e = edges[edgeIdx];
            if ((e.nodeA == i && e.nodeB == j) || (e.nodeA == j && e.nodeB == i)) {
                return e.matchCount;
            }
        }
        return 0;
    }
};

#endif // SFMTYPES_H