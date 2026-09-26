#pragma once

#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "OnnxHelper.h"


/**
 * SuperPoint inference with onnxruntime.
 *
 * The ONNX model only contains the dense (fully convolutional) part of SuperPoint:
 *   input  : "image"       (1, 1, H, W)      float32, value range [0, 1], NCHW
 *   output : "descriptors" (1, H/8, W/8, 256) float32, L2-normalized along channel, NHWC
 *            "scores"      (1, H/8, W/8, 65)  float32, raw logits (softmax is done here), NHWC
 *
 * Everything after that -- softmax, depth-to-space, NMS, thresholding, top-K and
 * descriptor sampling -- is implemented here, mirroring xfeatc's structure.
 */
class SuperPoint {
public:
    struct Config {
        // SuperPoint default: nms_radius = 4  ->  kernel = 2*4+1 = 9.
        // Do NOT reuse xfeat's 5: it would change the number of points drastically.
        int   nmsKernelSize   = 9;
        // SuperPoint default detection_threshold (xfeat uses 0.05).
        float scoreThresh     = 0.005f;
        // 0 or negative = keep all
        int   maxKeypoints    = 1000;
        // pad the image up to a multiple of this (must be a multiple of 8)
        int   padMultiple     = 32;
        // Measured on an 8-thread CPU @640x640: 1 -> 0.84s, 4 -> 0.36s, 8 -> 0.38s.
        int   intraThreads    = 4;
        int   interThreads    = 1;
        bool  useCuda         = false;   // needs a GPU build of onnxruntime + SUPERPOINT_USE_CUDA
        int   cudaDeviceId    = 0;
        // true  -> std::exp, exact but slower (use for alignment checks)
        // false -> FastExp bit-trick approximation (xfeatc style)
        bool  preciseSoftmax  = false;
        // NMS strategy:
        //   2 (default) - faithful port of SuperPoint's batched_nms (2 suppression
        //                 rounds). Matches the pytorch reference point-for-point.
        //   1           - single-pass local maximum via cv::dilate (fastest).
        //   0           - xfeatc style mask-based single pass: only one point is kept
        //                 per window, so it produces ~15% fewer keypoints than the
        //                 reference (all of them still valid, just more spread out).
        int   nmsMode         = 2;
    };

    struct ScoredPoint {
        int x;
        int y;
        float score;
    };

    explicit SuperPoint(const std::string &modelFile, const Config &cfg = Config());

    /**
     * @param img   grayscale or BGR image
     * @param keys  integer pixel coordinates (SuperPoint has no sub-pixel refinement)
     * @param descs N x 256, CV_32F, L2-normalized
     */
    void DetectAndCompute(const cv::Mat &img, std::vector<cv::KeyPoint> &keys, cv::Mat &descs);

    /// Dump raw network outputs (before post-processing) for cross-checking against pytorch.
    void EnableDump(const std::string &dir) { dumpDir_ = dir; }

    int PaddedHeight() const { return Hp_; }
    int PaddedWidth() const { return Wp_; }

private:
    void Preprocess(const cv::Mat &img, cv::Mat &padded);

    void SoftmaxScore(float *score, int h, int w, int c);

    void FlattenScore(const float *src, float *dst);

    void Nms(const cv::Mat &scores, float scoreThresh, int kernelSize, std::vector<ScoredPoint> &points);

    /// Faithful port of superpoint_pytorch.batched_nms().
    void NmsBatched(const cv::Mat &scores, float scoreThresh, int kernelSize,
                    std::vector<ScoredPoint> &points);

    void InterpDescriptor(const float *descMat, float *descriptor, float ptx, float pty);

    void DumpRaw(const float *descPtr, const float *scorePtr);

    void DumpScoreMap(const cv::Mat &scoreImg);

    void DumpResult(const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs);

    void WriteFloats(const std::string &name, const float *data, size_t count);

private:
    Config cfg_;

    std::unique_ptr<Ort::Env> ortEnv_;
    std::unique_ptr<Ort::Session> ortSession_;

    std::vector<TensorInfo> inputInfos_, outputInfos_;
    std::vector<std::string> inputNamesStr_, outputNamesStr_;
    std::vector<const char *> inputNames_, outputNames_;

    int inputC_   = 1;    // static, read from the model
    int descDim_  = 256;  // static, read from the model
    int scoreCh_  = 65;   // static, read from the model
    int descIdx_  = 0;    // index of "descriptors" in the output list (found by name)
    int scoreIdx_ = 1;    // index of "scores"

    // Runtime geometry. NOTE: with dynamic axes the model reports -1 for H/W,
    // so these are maintained here instead of being read from inputInfos_.
    int origH_ = 0, origW_ = 0;   // original (un-padded) image size
    int Hp_ = 0, Wp_ = 0;         // padded network input size
    int Hd8_ = 0, Wd8_ = 0;       // Hp_/8, Wp_/8

    std::vector<ScoredPoint> scoredPoints_;
    std::string dumpDir_;
};
