#include "SuperPoint.h"
#include "Timer.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

// https://gist.github.com/jrade/293a73f89dfef51da6522428c857802d
inline float FastExp(float x) {
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

} // namespace


SuperPoint::SuperPoint(const std::string &modelFile, const Config &cfg) : cfg_(cfg) {
    std::vector<ORTCHAR_T> modelFileOrt;
    OnnxHelper::Str2Ort(modelFile, modelFileOrt);

    ortEnv_ = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint"));

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(cfg_.intraThreads);
    sessionOptions.SetInterOpNumThreads(cfg_.interThreads);
    // SuperPoint is a plain CNN, so full graph optimization is safe here.
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef SUPERPOINT_USE_CUDA
    if (cfg_.useCuda) {
        try {
            OrtCUDAProviderOptions cudaOpts{};
            cudaOpts.device_id = cfg_.cudaDeviceId;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOpts);
            std::cout << "[SuperPoint] CUDA ExecutionProvider enabled (device "
                      << cfg_.cudaDeviceId << ")" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[SuperPoint] CUDA unavailable, falling back to CPU: " << e.what() << std::endl;
            cfg_.useCuda = false;
        }
    }
#else
    if (cfg_.useCuda) {
        std::cerr << "[SuperPoint] CUDA requested but this build has no CUDA support "
                     "(reconfigure with -DUSE_CUDA=ON). Using CPU." << std::endl;
        cfg_.useCuda = false;
    }
#endif

    ortSession_ = std::unique_ptr<Ort::Session>(
            new Ort::Session(*ortEnv_, modelFileOrt.data(), sessionOptions));

    OnnxHelper::GetModelInfo(*ortSession_, inputInfos_, outputInfos_);

    inputNamesStr_  = ortSession_->GetInputNames();
    outputNamesStr_ = ortSession_->GetOutputNames();
    for (auto &s : inputNamesStr_)  inputNames_.push_back(s.c_str());
    for (auto &s : outputNamesStr_) outputNames_.push_back(s.c_str());

    // Static dimensions are trustworthy even with dynamic axes.
    if (inputInfos_[0].shape.size() == 4) {
        inputC_ = (int)inputInfos_[0].shape[1];
    }

    // Locate outputs by name instead of hard-coded indices.
    auto findOutput = [&](const std::string &key) -> int {
        for (size_t i = 0; i < outputNamesStr_.size(); ++i) {
            if (outputNamesStr_[i].find(key) != std::string::npos) return (int)i;
        }
        return -1;
    };
    int di = findOutput("descriptors");
    int si = findOutput("scores");
    if (di >= 0) descIdx_ = di;
    if (si >= 0) scoreIdx_ = si;

    const auto &dShape = outputInfos_[descIdx_].shape;
    descDim_ = (int)dShape[dShape.size() - 1];          // NHWC -> last dim
    const auto &sShape = outputInfos_[scoreIdx_].shape;
    if (sShape.size() == 4) {
        scoreCh_ = (int)sShape[sShape.size() - 1];      // logits variant: 65
    } else {
        std::cerr << "[SuperPoint] unsupported scores layout (expected NHWC logits)" << std::endl;
        exit(-1);
    }

    OnnxHelper::PrintModelInfo(inputInfos_, outputInfos_);
}


void SuperPoint::Preprocess(const cv::Mat &img, cv::Mat &padded) {
    cv::Mat gray;
    if (img.channels() == 3) {
        // OpenCV BGR2GRAY uses 0.114B + 0.587G + 0.299R, identical to torch's [0.299,0.587,0.114].
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = img;
    }

    origH_ = gray.rows;
    origW_ = gray.cols;

    // Pad to a multiple of padMultiple by appending zeros at the right/bottom.
    // Zero padding (instead of resize) keeps pixel values and scale untouched, so
    // keypoint coordinates need no correction afterwards.
    const int m = cfg_.padMultiple > 0 ? cfg_.padMultiple : 8;
    Hp_ = ((origH_ + m - 1) / m) * m;
    Wp_ = ((origW_ + m - 1) / m) * m;
    Hd8_ = Hp_ / 8;
    Wd8_ = Wp_ / 8;

    cv::Mat f32;
    gray.convertTo(f32, CV_32F, 1.0f / 255.0f);   // SuperPoint expects [0, 1]

    if (Hp_ == origH_ && Wp_ == origW_) {
        padded = f32;
    } else {
        cv::copyMakeBorder(f32, padded, 0, Hp_ - origH_, 0, Wp_ - origW_,
                           cv::BORDER_CONSTANT, cv::Scalar(0.f));
    }
    CV_Assert(padded.isContinuous());
}


void SuperPoint::DetectAndCompute(const cv::Mat &img, std::vector<cv::KeyPoint> &keys, cv::Mat &descs) {
    Timer timer;
    keys.clear();

    timer.Reset();
    cv::Mat padded;
    Preprocess(img, padded);
    const double t_pre = timer.Elapse();

    // Dynamic shape: build the input shape from the actual padded size every time.
    // (Reading it from inputInfos_ would give -1.)
    const std::vector<int64_t> inShape = {1, (int64_t)inputC_, (int64_t)Hp_, (int64_t)Wp_};
    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(OnnxHelper::CreateTensor<float>(
            inShape, padded.ptr<float>(), (size_t)Hp_ * Wp_, true));

    timer.Reset();
    auto outputTensors = ortSession_->Run(Ort::RunOptions{nullptr}, inputNames_.data(),
                                          inputTensors.data(), 1,
                                          outputNames_.data(), outputNames_.size());
    const double t_infer = timer.Elapse();

    float *descPtr  = outputTensors[descIdx_].GetTensorMutableData<float>();   // (1,Hd8,Wd8,256)
    float *scorePtr = outputTensors[scoreIdx_].GetTensorMutableData<float>();  // (1,Hd8,Wd8,65)

    if (!dumpDir_.empty()) DumpRaw(descPtr, scorePtr);

    timer.Reset();
    SoftmaxScore(scorePtr, Hd8_, Wd8_, scoreCh_);
    const double t_softmax = timer.Elapse();

    timer.Reset();
    cv::Mat scoreImg(Hp_, Wp_, CV_32F);
    FlattenScore(scorePtr, scoreImg.ptr<float>());
    const double t_flatten = timer.Elapse();
    if (!dumpDir_.empty()) DumpScoreMap(scoreImg);

    timer.Reset();
    Nms(scoreImg, cfg_.scoreThresh, cfg_.nmsKernelSize, scoredPoints_);
    const double t_nms = timer.Elapse();

    timer.Reset();
    std::sort(scoredPoints_.begin(), scoredPoints_.end(),
              [](const ScoredPoint &a, const ScoredPoint &b) { return a.score > b.score; });
    const double t_sort = timer.Elapse();

    // Nms already skips a border of kernelSize/2 (=4) pixels, which is exactly
    // SuperPoint's remove_borders=4. Additionally drop anything inside the padded region.
    const int border = cfg_.nmsKernelSize / 2;
    const int xMin = border, yMin = border;
    const int xMax = std::max(origW_ - border, xMin + 1);
    const int yMax = std::max(origH_ - border, yMin + 1);

    for (const auto &pt : scoredPoints_) {
        if (pt.x < xMin || pt.x >= xMax || pt.y < yMin || pt.y >= yMax) continue;
        if (cfg_.maxKeypoints > 0 && (int)keys.size() >= cfg_.maxKeypoints) break;
        keys.emplace_back((float)pt.x, (float)pt.y, 1.0f, -1.0f, pt.score);
    }

    timer.Reset();
    descs = cv::Mat::zeros((int)keys.size(), descDim_, CV_32F);
    const float sx = (float)Wd8_ / (float)Wp_;
    const float sy = (float)Hd8_ / (float)Hp_;
    for (int n = 0; n < (int)keys.size(); ++n) {
        // Equivalent of grid_sample(bilinear, align_corners=False) used by
        // superpoint_pytorch.sample_descriptors().
        const float px = (keys[n].pt.x + 0.5f) * sx - 0.5f;
        const float py = (keys[n].pt.y + 0.5f) * sy - 0.5f;
        InterpDescriptor(descPtr, descs.ptr<float>(n), px, py);
    }
    const double t_interp = timer.Elapse();

    if (!dumpDir_.empty()) DumpResult(keys, descs);

    std::cout << "pre=" << t_pre << "s, infer=" << t_infer << "s, softmax=" << t_softmax
              << "s, flatten=" << t_flatten << "s, nms=" << t_nms << "s, sort=" << t_sort
              << "s, interp=" << t_interp << "s, keys=" << keys.size() << std::endl;
}


void SuperPoint::SoftmaxScore(float *score, int h, int w, int c) {
    for (int i = 0; i < h * w; ++i) {
        float *ptr = score + i * c;

        // Subtracting the max keeps FastExp accurate (torch does the same).
        float maxVal = ptr[0];
        for (int k = 1; k < c; ++k) maxVal = std::max(maxVal, ptr[k]);

        float sum = 0.f;
        for (int k = 0; k < c; ++k) {
            float e = cfg_.preciseSoftmax ? std::exp(ptr[k] - maxVal) : FastExp(ptr[k] - maxVal);
            ptr[k] = e;
            sum += e;
        }
        float invSum = 1.0f / std::max(sum, 1e-12f);
        for (int k = 0; k < c; ++k) ptr[k] *= invSum;
    }
}


void SuperPoint::FlattenScore(const float *src, float *dst) {
    // Identical to XFeat::FlattenScore: the 64 NHWC channels of a cell are laid out
    // row-major as an 8x8 block (equivalent to nn.PixelShuffle(8) / DepthToSpace CRD).
    // The 65th channel is the dustbin and is dropped.
    for (int i = 0; i < Hd8_; ++i) {
        for (int j = 0; j < Wd8_; ++j) {
            const float *srcPtr = src + (i * Wd8_ + j) * scoreCh_;
            float *dstPtr = dst + (i * 8) * Wp_ + (j * 8);
            for (int k = 0; k < 8; ++k) {
                for (int l = 0; l < 8; ++l) {
                    dstPtr[k * Wp_ + l] = srcPtr[k * 8 + l];
                }
            }
        }
    }
}


void SuperPoint::Nms(const cv::Mat &scores, float scoreThresh, int kernelSize,
                     std::vector<ScoredPoint> &points) {
    points.clear();

    if (cfg_.nmsMode == 2) {
        NmsBatched(scores, scoreThresh, kernelSize, points);
        return;
    }

    if (cfg_.nmsMode == 1) {
        // local maximum == value not smaller than the dilation over the window
        cv::Mat dilated;
        cv::dilate(scores, dilated, cv::getStructuringElement(
                cv::MORPH_RECT, cv::Size(kernelSize, kernelSize)));
        const int half = kernelSize / 2;
        for (int y = half; y < scores.rows - half; ++y) {
            const float *sRow = scores.ptr<float>(y);
            const float *dRow = dilated.ptr<float>(y);
            for (int x = half; x < scores.cols - half; ++x) {
                if (sRow[x] > scoreThresh && sRow[x] >= dRow[x]) {
                    points.push_back({x, y, sRow[x]});
                }
            }
        }
        return;
    }

    // xfeatc style: single pass with a suppression mask (O(1) neighbourhood test)
    const int rows = scores.rows;
    const int cols = scores.cols;
    const int halfKernelSize = kernelSize / 2;
    cv::Mat mask = cv::Mat::ones(rows, cols, CV_8U);
    const float *scorePtr = scores.ptr<float>();
    uchar *maskPtr = mask.ptr<uchar>();

    std::vector<int> ptrOffsets;
    ptrOffsets.reserve((size_t)kernelSize * kernelSize);
    for (int i = -halfKernelSize; i <= halfKernelSize; ++i) {
        for (int j = -halfKernelSize; j <= halfKernelSize; ++j) {
            if (i == 0 && j == 0) continue;
            ptrOffsets.push_back(i * cols + j);
        }
    }

    for (int i = halfKernelSize; i < rows - halfKernelSize; ++i) {
        for (int j = halfKernelSize; j < cols - halfKernelSize; ++j) {
            const int addr = i * cols + j;
            if (maskPtr[addr] == 0) continue;

            const float score = scorePtr[addr];
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
                for (const auto &offset : ptrOffsets) maskPtr[addr + offset] = 0;
            }
        }
    }
}


void SuperPoint::NmsBatched(const cv::Mat &scores, float scoreThresh, int kernelSize,
                            std::vector<ScoredPoint> &points) {
    // Port of:
    //   def max_pool(x): return F.max_pool2d(x, kernel_size=2r+1, stride=1, padding=r)
    //   max_mask = scores == max_pool(scores)
    //   for _ in range(2):
    //       supp_mask = max_pool(max_mask.float()) > 0
    //       supp_scores = where(supp_mask, zeros, scores)
    //       new_max_mask = supp_scores == max_pool(supp_scores)
    //       max_mask = max_mask | (new_max_mask & ~supp_mask)
    //
    // max_pool2d pads with -inf; cv::dilate replicates the border, and replication can
    // never introduce a value larger than an existing one, so the result is identical.
    points.clear();

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    const int rows = scores.rows, cols = scores.cols;

    auto toMask01 = [](cv::Mat &m) { m.convertTo(m, CV_32F, 1.0 / 255.0); };

    cv::Mat pooled;
    cv::dilate(scores, pooled, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);

    cv::Mat maxMask;                                   // 1.0 at local maxima
    cv::compare(scores, pooled, maxMask, cv::CMP_EQ);
    toMask01(maxMask);

    for (int iter = 0; iter < 2; ++iter) {
        cv::Mat suppMask;                              // dilated max_mask, 1.0 in the neighbourhood of a maximum
        cv::dilate(maxMask, suppMask, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);

        cv::Mat suppScores(rows, cols, CV_32F);
        for (int y = 0; y < rows; ++y) {
            const float *s = scores.ptr<float>(y);
            const float *sm = suppMask.ptr<float>(y);
            float *d = suppScores.ptr<float>(y);
            for (int x = 0; x < cols; ++x) d[x] = (sm[x] > 0.f) ? 0.f : s[x];
        }

        cv::Mat newPooled, newMax;
        cv::dilate(suppScores, newPooled, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        cv::compare(suppScores, newPooled, newMax, cv::CMP_EQ);
        toMask01(newMax);

        // max_mask |= new_max_mask & ~supp_mask
        for (int y = 0; y < rows; ++y) {
            float *mm = maxMask.ptr<float>(y);
            const float *nm = newMax.ptr<float>(y);
            const float *sm = suppMask.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                if (nm[x] > 0.f && !(sm[x] > 0.f)) mm[x] = 1.0f;
            }
        }
    }

    // SuperPoint then applies remove_borders (= nms_radius) and keeps scores > threshold.
    const int half = kernelSize / 2;
    for (int y = half; y < rows - half; ++y) {
        const float *s = scores.ptr<float>(y);
        const float *mm = maxMask.ptr<float>(y);
        for (int x = half; x < cols - half; ++x) {
            if (mm[x] > 0.f && s[x] > scoreThresh) points.push_back({x, y, s[x]});
        }
    }
}


void SuperPoint::InterpDescriptor(const float *descMat, float *descriptor, float ptx, float pty) {
    const int x0 = cvFloor(ptx);
    const int y0 = cvFloor(pty);
    const float dx = ptx - static_cast<float>(x0);
    const float dy = pty - static_cast<float>(y0);
    const int C = descDim_;

    // grid_sample default padding_mode is 'zeros': out-of-range taps contribute
    // nothing. Do NOT clamp to the border -- that changes the result.
    auto inside = [&](int xi, int yi) {
        return xi >= 0 && yi >= 0 && xi < Wd8_ && yi < Hd8_;
    };
    auto at = [&](int xi, int yi) -> const float * {
        return inside(xi, yi) ? descMat + ((yi * Wd8_ + xi) * C) : nullptr;
    };

    const float w00 = (1.f - dx) * (1.f - dy);
    const float w10 = dx * (1.f - dy);
    const float w01 = (1.f - dx) * dy;
    const float w11 = dx * dy;

    const float *p00 = at(x0,     y0);
    const float *p10 = at(x0 + 1, y0);
    const float *p01 = at(x0,     y0 + 1);
    const float *p11 = at(x0 + 1, y0 + 1);

    double sum = 0;
    for (int c = 0; c < C; ++c) {
        float v = 0.f;
        if (p00) v += w00 * p00[c];
        if (p10) v += w10 * p10[c];
        if (p01) v += w01 * p01[c];
        if (p11) v += w11 * p11[c];
        descriptor[c] = v;
        sum += (double)v * v;
    }

    // sample_descriptors() L2-normalizes again after interpolation
    const float invNorm = static_cast<float>(1.0 / std::max(std::sqrt(sum), 1e-12));
    for (int c = 0; c < C; ++c) descriptor[c] *= invNorm;
}


void SuperPoint::WriteFloats(const std::string &name, const float *data, size_t count) {
    const std::string path = dumpDir_ + "/" + name;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[SuperPoint] cannot write " << path << std::endl;
        return;
    }
    ofs.write(reinterpret_cast<const char *>(data), (std::streamsize)(count * sizeof(float)));
    std::cout << "[dump] " << path << " (" << count << " floats)" << std::endl;
}


void SuperPoint::DumpRaw(const float *descPtr, const float *scorePtr) {
    // Raw network outputs go out BEFORE softmax, so the python side can separate
    // "network accuracy" from "post-processing accuracy".
    WriteFloats("cpp_logits.bin", scorePtr, (size_t)Hd8_ * Wd8_ * scoreCh_);
    WriteFloats("cpp_desc.bin", descPtr, (size_t)Hd8_ * Wd8_ * descDim_);

    // geometry file so the python side knows how to reshape
    std::ofstream g(dumpDir_ + "/cpp_shape.txt");
    g << Hp_ << " " << Wp_ << " " << Hd8_ << " " << Wd8_ << " "
      << descDim_ << " " << scoreCh_ << " " << origW_ << " " << origH_ << std::endl;
}


void SuperPoint::DumpScoreMap(const cv::Mat &scoreImg) {
    WriteFloats("cpp_scoremap.bin", scoreImg.ptr<float>(), (size_t)Hp_ * Wp_);
}


void SuperPoint::DumpResult(const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs) {
    std::ofstream kf(dumpDir_ + "/cpp_kpts.txt");
    for (const auto &k : keys) {
        // float precision is enough for an integer coordinate
        kf << k.pt.x << " " << k.pt.y << " " << k.response << "\n";
    }
    std::cout << "[dump] " << dumpDir_ << "/cpp_kpts.txt (" << keys.size() << " points)" << std::endl;

    if (!descs.empty()) {
        WriteFloats("cpp_kpts_desc.bin", descs.ptr<float>(), (size_t)descs.total());
    }
}
