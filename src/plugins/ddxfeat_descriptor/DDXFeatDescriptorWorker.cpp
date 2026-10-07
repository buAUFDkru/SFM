/**
 * @file DDXFeatDescriptorWorker.cpp
 * @brief DD-XFeat 描述符 Worker（基于ONNX Runtime推理）
 *
 * DD-XFeat 为 XFeat 的改进卷积神经网络版本（模型来自
 * D:\project\python\accelerated_features-main 训练导出）：
 * 仅 CNN 主干不同，后处理逻辑与 XFeat 完全一致。
 *
 * 模型目录：<可执行程序目录>/models（外部目录）
 *
 * 核心流程：
 * 1. 根据输入图像高度选择最接近的外部 ONNX 模型
 * 2. 将图像 resize 到模型输入尺寸
 * 3. 归一化后送入 ONNX 推理
 * 4. 对输出进行 softmax、NMS、描述符插值
 * 5. 将关键点坐标映射回原图尺寸
 */

#include "DDXFeatDescriptorWorker.h"

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QRegularExpression>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <opencv2/imgproc.hpp>

// ═════════════════════════════════════════════════════════════════════════════
//  内联辅助函数（与 XFeat 一致，参考 xfeatc-main/XFeat.cc）
// ═════════════════════════════════════════════════════════════════════════════

/// 双三次插值权重计算
inline void calcBicubicWeights(float t, float &wm1, float &w0, float &w1, float &w2)
{
    constexpr float a = -0.75f;
    float t2 = t * t;
    float t3 = t2 * t;
    wm1 = a * (t3 - 2 * t2 + t);
    w0  = (a + 2) * t3 - (a + 3) * t2 + 1;
    w1  = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t;
    w2  = a * (-t3 + t2);
}

/// 快速指数函数近似
inline float fastExp(float x)
{
    constexpr float a = (1 << 23) / 0.69314718f;
    constexpr float b = (1 << 23) * (127 - 0.043677448f);
    x = a * x + b;
    constexpr float c = (1 << 23);
    constexpr float d = (1 << 23) * 255;
    if (x < c || x > d)
        x = (x < c) ? 0.0f : d;
    uint32_t n = static_cast<uint32_t>(x);
    memcpy(&x, &n, 4);
    return x;
}

// ═════════════════════════════════════════════════════════════════════════════
//  构造/析构
// ═════════════════════════════════════════════════════════════════════════════

DDXFeatDescriptorWorker::DDXFeatDescriptorWorker(const QJsonObject &params)
{
    if (params.contains(QStringLiteral("scoreThreshold")))
        m_scoreThreshold = static_cast<float>(
            params.value(QStringLiteral("scoreThreshold")).toDouble(0.05));
    if (params.contains(QStringLiteral("nmsKernelSize")))
        m_nmsKernelSize = params.value(QStringLiteral("nmsKernelSize")).toInt(5);
    if (params.contains(QStringLiteral("maxFeatures")))
        m_maxFeatures = params.value(QStringLiteral("maxFeatures")).toInt(8192);

    m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DD-XFeat");
    // 延迟扫描模型，避免在 QCoreApplication 初始化前调用 applicationDirPath()
}

DDXFeatDescriptorWorker::~DDXFeatDescriptorWorker() = default;

// ═════════════════════════════════════════════════════════════════════════════
//  IDescriptorWorker 接口实现
// ═════════════════════════════════════════════════════════════════════════════

FeatureResult DDXFeatDescriptorWorker::extract(const ImageEntry &image)
{
    if (image.imageGray.empty())
        return {};

    if (!loadModelForHeight(image.imageGray.rows)) {
        qWarning() << "[DD-XFeat] 无法加载匹配图像高度的模型";
        return {};
    }

    FeatureResult r;
    r.imageId = image.id;
    r.imageWidth = image.imageGray.cols;
    r.imageHeight = image.imageGray.rows;
    if (!detectAndCompute(image.imageGray, r))
        return {};
    r.elapsedMs = 0.0;
    return r;
}

// ═════════════════════════════════════════════════════════════════════════════
//  模型管理
// ═════════════════════════════════════════════════════════════════════════════

void DDXFeatDescriptorWorker::scanAvailableModels()
{
    m_availableModels.clear();

    // 扫描可执行程序目录下的 models 目录（外部目录）
    QDir modelsDir(QCoreApplication::applicationDirPath() + QStringLiteral("/models"));
    if (modelsDir.exists()) {
        QStringList files = modelsDir.entryList({ QStringLiteral("ddxfeat*.onnx") },
                                                QDir::Files, QDir::Name);
        for (const QString &file : files) {
            QRegularExpression re(QStringLiteral("ddxfeat(\\d+)x(\\d+)\\.onnx"),
                                  QRegularExpression::CaseInsensitiveOption);
            QRegularExpressionMatch match = re.match(file);
            if (match.hasMatch()) {
                int h = match.captured(1).toInt();
                QString filePath = modelsDir.filePath(file);
                if (!m_availableModels.contains(h))
                    m_availableModels[h] = filePath;
            }
        }
    }

    if (m_availableModels.isEmpty()) {
        qWarning() << "[DD-XFeat] 警告: 未在" << modelsDir.absolutePath()
                   << "下找到任何 ddxfeat*.onnx 模型文件";
    }
}

bool DDXFeatDescriptorWorker::loadModelForHeight(int imageHeight)
{
    // 首次调用时扫描可用模型
    if (m_availableModels.isEmpty())
        scanAvailableModels();

    if (m_availableModels.isEmpty()) {
        qWarning() << "[DD-XFeat] 无可用模型";
        return false;
    }

    // 找到与图像高度最接近的模型
    int bestH = -1;
    int minDiff = std::numeric_limits<int>::max();
    for (auto it = m_availableModels.begin(); it != m_availableModels.end(); ++it) {
        int diff = std::abs(it.key() - imageHeight);
        if (diff < minDiff) {
            minDiff = diff;
            bestH = it.key();
        }
    }

    if (bestH < 0)
        return false;

    // 如果已经是当前加载的模型，直接返回
    if (m_ortSession && bestH == m_modelH)
        return true;

    // 加载新模型
    QString modelPath = m_availableModels[bestH];

    // 将 QString 转为 wchar_t 数组（Windows 上 ORTCHAR_T = wchar_t）
    std::wstring wsModelPath = modelPath.toStdWString();
    std::vector<ORTCHAR_T> modelFileOrt(wsModelPath.begin(), wsModelPath.end());
    modelFileOrt.push_back(ORTCHAR_T('\0'));

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    try {
        m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, modelFileOrt.data(), sessionOptions);
    } catch (const Ort::Exception &e) {
        qWarning() << "[DD-XFeat] ONNX加载失败:" << e.what();
        m_ortSession.reset();
        return false;
    }

    // 获取输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;

    m_inputNames.clear();
    m_outputNames.clear();
    m_inputNameStorage.clear();
    m_outputNameStorage.clear();

    size_t numInputs = m_ortSession->GetInputCount();
    m_inputNameStorage.reserve(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        std::string name(m_ortSession->GetInputNameAllocated(i, allocator).get());
        auto typeInfo = m_ortSession->GetInputTypeInfo(i);
        auto shape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (i == 0)
            m_inputShape = shape;
        m_inputNameStorage.push_back(std::move(name));
    }
    for (const auto &s : m_inputNameStorage)
        m_inputNames.push_back(s.c_str());

    size_t numOutputs = m_ortSession->GetOutputCount();
    m_outputNameStorage.reserve(numOutputs);
    for (size_t i = 0; i < numOutputs; ++i) {
        std::string name(m_ortSession->GetOutputNameAllocated(i, allocator).get());
        m_outputNameStorage.push_back(std::move(name));
    }
    for (const auto &s : m_outputNameStorage)
        m_outputNames.push_back(s.c_str());

    // 解析模型输入尺寸
    m_modelH = static_cast<int>(m_inputShape[2]);
    m_modelW = static_cast<int>(m_inputShape[3]);
    m_modelHd8 = m_modelH / 8;
    m_modelWd8 = m_modelW / 8;

    // 解析描述符通道数：输出 0 形状 [1, H/8, W/8, C]，
    // DD-XFeat 为 128 维（XFeat 64 维），按模型实际导出解析
    auto out0Info = m_ortSession->GetOutputTypeInfo(0);
    auto out0Shape = out0Info.GetTensorTypeAndShapeInfo().GetShape();
    if (out0Shape.size() == 4 && out0Shape[3] > 0)
        m_descDim = static_cast<int>(out0Shape[3]);

    if (m_modelH % 32 != 0 || m_modelW % 32 != 0) {
        qWarning() << "[DD-XFeat] 模型输入尺寸必须能被32整除";
        m_ortSession.reset();
        return false;
    }

    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  推理核心
// ═════════════════════════════════════════════════════════════════════════════

bool DDXFeatDescriptorWorker::detectAndCompute(const cv::Mat &image, FeatureResult &outResult)
{
    if (!m_ortSession)
        return false;

    int origH = image.rows;
    int origW = image.cols;

    // ── 将图像 resize 到模型输入尺寸 ──
    cv::Mat resizedImg;
    if (origH != m_modelH || origW != m_modelW) {
        cv::resize(image, resizedImg, cv::Size(m_modelW, m_modelH));
    } else {
        resizedImg = image.clone();
    }

    // ── 转为 float 并归一化 ──
    cv::Mat fimg;
    resizedImg.convertTo(fimg, CV_32F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(fimg, mean, stddev);
    fimg -= mean;
    fimg /= stddev;

    // ── 创建输入 tensor ──
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, fimg.ptr<float>(), m_modelW * m_modelH,
        m_inputShape.data(), m_inputShape.size()));

    // ── 运行推理 ──
    std::vector<Ort::Value> outputTensors;
    try {
        outputTensors = m_ortSession->Run(
            Ort::RunOptions{nullptr}, m_inputNames.data(),
            inputTensors.data(), 1, m_outputNames.data(), m_outputNames.size());
    } catch (const Ort::Exception &e) {
        qWarning() << "[DD-XFeat] 推理失败:" << e.what();
        return false;
    }

    // ── 输出解析（与 XFeat 一致） ──
    // output 0: [1, H/8, W/8, 64] 描述符
    // output 1: [1, H/8, W/8, 65] 关键点得分
    // output 2: [1, 1, H/8, W/8] 可靠性热图
    auto *kptScorePtr = outputTensors[1].GetTensorMutableData<float>();
    const int shw = m_modelHd8 * m_modelWd8;

    // softmax
    softmaxScore(kptScorePtr, m_modelHd8, m_modelWd8, 65);

    // 将 65 通道得分展平为 H×W 得分图
    cv::Mat scoreImg(m_modelH, m_modelW, CV_32F);
    flattenScore(kptScorePtr, scoreImg.ptr<float>());

    // NMS
    std::vector<ScoredPoint> scoredPoints;
    nms(scoreImg, m_scoreThreshold, m_nmsKernelSize, scoredPoints);

    // 可靠性热图
    auto *heatMapPtr = outputTensors[2].GetTensorMutableData<float>();
    cv::Mat heatMapSmall(m_modelHd8, m_modelWd8, CV_32F, heatMapPtr);
    cv::Mat heatMapFull;
    cv::resize(heatMapSmall, heatMapFull, cv::Size(m_modelW, m_modelH));

    // 得分 × 可靠性
    for (auto &pt : scoredPoints) {
        pt.score *= heatMapFull.at<float>(pt.y, pt.x);
    }

    // 按得分排序
    std::sort(scoredPoints.begin(), scoredPoints.end(),
              [](const ScoredPoint &a, const ScoredPoint &b) {
                  return a.score > b.score;
              });

    // 转换为 cv::KeyPoint，按 maxFeatures 限制数量，过滤边缘点
    const int maxKeypoints = m_maxFeatures;
    const int minEdge = 12;
    std::vector<cv::KeyPoint> keys;
    if (maxKeypoints > 0)
        keys.reserve(maxKeypoints);
    for (const auto &pt : scoredPoints) {
        if (pt.score <= 0)
            break; // 已排序，后续 score 都不大于 0
        if (pt.x <= minEdge || pt.x >= m_modelW - minEdge ||
            pt.y <= minEdge || pt.y >= m_modelH - minEdge) {
            continue;
        }
        keys.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y), 0);
        if (maxKeypoints > 0 && static_cast<int>(keys.size()) >= maxKeypoints)
            break;
    }

    // ── 描述符提取 ──
    auto *descTensorPtr = outputTensors[0].GetTensorMutableData<float>();

    // 归一化描述符（沿通道维度，通道数 m_descDim）
    for (int i = 0; i < shw; ++i) {
        double sum = 0;
        float *ptr = &descTensorPtr[i * m_descDim];
        for (int j = 0; j < m_descDim; ++j) {
            sum += ptr[j] * ptr[j];
        }
        float invNorm = static_cast<float>(1.0 / std::max(std::sqrt(sum), 1e-12));
        for (int j = 0; j < m_descDim; ++j) {
            ptr[j] *= invNorm;
        }
    }

    // 双三次插值获取每个关键点的描述符
    cv::Mat descs = cv::Mat::zeros(static_cast<int>(keys.size()), m_descDim, CV_32F);
    const float widthScale = static_cast<float>(m_modelWd8) / static_cast<float>(m_modelW - 1);
    const float heightScale = static_cast<float>(m_modelHd8) / static_cast<float>(m_modelH - 1);

    for (int n = 0; n < static_cast<int>(keys.size()); ++n) {
        float x = keys[n].pt.x * widthScale - 0.5f;
        float y = keys[n].pt.y * heightScale - 0.5f;
        interpDescriptor(descTensorPtr, descs.ptr<float>(n), x, y);
    }

    // ── 将关键点坐标映射回原图尺寸 ──
    float scaleX = static_cast<float>(origW) / static_cast<float>(m_modelW);
    float scaleY = static_cast<float>(origH) / static_cast<float>(m_modelH);
    for (auto &key : keys) {
        key.pt.x *= scaleX;
        key.pt.y *= scaleY;
    }

    outResult.keypoints = std::move(keys);
    outResult.descriptors = std::move(descs);

    return !outResult.keypoints.empty();
}

// ═════════════════════════════════════════════════════════════════════════════
//  内部辅助函数
// ═════════════════════════════════════════════════════════════════════════════

void DDXFeatDescriptorWorker::softmaxScore(float *score, int h, int w, int c)
{
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float *ptr = score + i * w * c + j * c;
            float sum = 0;
            for (int k = 0; k < c; ++k) {
                float exp = fastExp(ptr[k]);
                ptr[k] = exp;
                sum += exp;
            }
            float invSum = 1.0f / sum;
            for (int k = 0; k < c; ++k) {
                ptr[k] *= invSum;
            }
        }
    }
}

void DDXFeatDescriptorWorker::flattenScore(float *src, float *dst)
{
    for (int i = 0; i < m_modelHd8; ++i) {
        for (int j = 0; j < m_modelWd8; ++j) {
            float *srcPtr = src + i * m_modelWd8 * 65 + j * 65;
            int iRow = i * 8;
            int jCol = j * 8;
            float *dstPtr = dst + iRow * m_modelW + jCol;
            for (int k = 0; k < 8; ++k) {
                for (int l = 0; l < 8; ++l) {
                    dstPtr[k * m_modelW + l] = srcPtr[k * 8 + l];
                }
            }
        }
    }
}

void DDXFeatDescriptorWorker::nms(const cv::Mat &scores, float scoreThresh, int kernelSize,
                                  std::vector<ScoredPoint> &points)
{
    points.clear();
    int rows = scores.rows;
    int cols = scores.cols;
    int halfKernel = kernelSize / 2;
    cv::Mat mask = cv::Mat::ones(rows, cols, CV_8U);
    const auto *scorePtr = scores.ptr<float>();
    auto *maskPtr = mask.ptr<uchar>();

    std::vector<int> ptrOffsets;
    ptrOffsets.reserve(kernelSize * kernelSize - 1);
    for (int i = -halfKernel; i <= halfKernel; ++i) {
        for (int j = -halfKernel; j <= halfKernel; ++j) {
            if (i == 0 && j == 0)
                continue;
            ptrOffsets.push_back(i * cols + j);
        }
    }

    for (int i = halfKernel; i < rows - halfKernel; ++i) {
        for (int j = halfKernel; j < cols - halfKernel; ++j) {
            int addr = i * cols + j;
            if (maskPtr[addr] == 0)
                continue;

            float score = scorePtr[addr];
            if (score <= scoreThresh) {
                maskPtr[addr] = 0;
                continue;
            }

            bool isMax = true;
            for (const auto &offset : ptrOffsets) {
                if (score < scorePtr[addr + offset]) {
                    maskPtr[addr] = 0;
                    isMax = false;
                    break;
                }
            }

            if (isMax) {
                points.push_back({j, i, score});
                for (const auto &offset : ptrOffsets) {
                    maskPtr[addr + offset] = 0;
                }
            }
        }
    }
}

void DDXFeatDescriptorWorker::interpDescriptor(const float *descMat, float *descriptor,
                                               float ptx, float pty)
{
    int x0 = cvFloor(ptx);
    int y0 = cvFloor(pty);
    int xm1 = x0 - 1;
    int ym1 = y0 - 1;
    float dx = ptx - static_cast<float>(x0);
    float dy = pty - static_cast<float>(y0);

    float wxm1, wx0, wx1, wx2;
    float wym1, wy0, wy1, wy2;
    calcBicubicWeights(dx, wxm1, wx0, wx1, wx2);
    calcBicubicWeights(dy, wym1, wy0, wy1, wy2);

    const float *desc_xm1_ym1 = descMat + ym1 * m_modelWd8 * m_descDim + xm1 * m_descDim;
    const float *desc_x0_ym1  = desc_xm1_ym1 + m_descDim;
    const float *desc_x1_ym1  = desc_x0_ym1 + m_descDim;
    const float *desc_x2_ym1  = desc_x1_ym1 + m_descDim;
    const float *desc_xm1_y0  = desc_xm1_ym1 + m_modelWd8 * m_descDim;
    const float *desc_x0_y0   = desc_xm1_y0 + m_descDim;
    const float *desc_x1_y0   = desc_x0_y0 + m_descDim;
    const float *desc_x2_y0   = desc_x1_y0 + m_descDim;
    const float *desc_xm1_y1  = desc_xm1_y0 + m_modelWd8 * m_descDim;
    const float *desc_x0_y1   = desc_xm1_y1 + m_descDim;
    const float *desc_x1_y1   = desc_x0_y1 + m_descDim;
    const float *desc_x2_y1   = desc_x1_y1 + m_descDim;
    const float *desc_xm1_y2  = desc_xm1_y1 + m_modelWd8 * m_descDim;
    const float *desc_x0_y2   = desc_xm1_y2 + m_descDim;
    const float *desc_x1_y2   = desc_x0_y2 + m_descDim;
    const float *desc_x2_y2   = desc_x1_y2 + m_descDim;

    double sum = 0;
    for (int i = 0; i < m_descDim; ++i) {
        float v_m1 = wxm1 * desc_xm1_ym1[i] + wx0 * desc_x0_ym1[i]
                    + wx1 * desc_x1_ym1[i] + wx2 * desc_x2_ym1[i];
        float v_0  = wxm1 * desc_xm1_y0[i] + wx0 * desc_x0_y0[i]
                    + wx1 * desc_x1_y0[i] + wx2 * desc_x2_y0[i];
        float v_1  = wxm1 * desc_xm1_y1[i] + wx0 * desc_x0_y1[i]
                    + wx1 * desc_x1_y1[i] + wx2 * desc_x2_y1[i];
        float v_2  = wxm1 * desc_xm1_y2[i] + wx0 * desc_x0_y2[i]
                    + wx1 * desc_x1_y2[i] + wx2 * desc_x2_y2[i];
        float v = wym1 * v_m1 + wy0 * v_0 + wy1 * v_1 + wy2 * v_2;
        descriptor[i] = v;
        sum += v * v;
    }

    float invNorm = static_cast<float>(1.0 / std::max(std::sqrt(sum), 1e-12));
    for (int i = 0; i < m_descDim; ++i) {
        descriptor[i] *= invNorm;
    }
}
