#include "fastbev_infer.h"
#include "bev_pool.h"
#include "postprocess.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <utility>
#include <chrono>
#include <atomic>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::milliseconds;

static double Elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        Clock::now() - start).count();
}

// ==================================================================
// Constants
// ==================================================================

static const float IMG_MEAN[3] = {123.675f, 116.28f, 103.53f};
static const float IMG_STD[3]  = {58.395f, 57.12f, 57.375f};

// ==================================================================
// Constructor
// ==================================================================

FastBEVInfer::FastBEVInfer(const Config& config) : config_(config)
{
    // Runtime GPU detection: if user requests GPU but no CUDA device is
    // available, automatically fall back to CPU for everything (DNN + Feature→BEV).
    use_gpu_ = config_.use_gpu;
    if (use_gpu_) {
        try {
            int ndev = cv::cuda::getCudaEnabledDeviceCount();
            if (ndev == 0) {
                printf("GPU requested but no CUDA devices found, falling back to CPU.\n");
                use_gpu_ = false;
            }
        } catch (const cv::Exception&) {
            printf("GPU requested but OpenCV was built without CUDA support, falling back to CPU.\n");
            use_gpu_ = false;
        }
    }

    printf("Loading 2D model: %s ...\n", config_.model_2d_path.c_str());
    net_2d_ = cv::dnn::readNetFromONNX(config_.model_2d_path);
    printf("2D model loaded.\n");

    printf("Loading 3D model: %s ...\n", config_.model_3d_path.c_str());
    net_3d_ = cv::dnn::readNetFromONNX(config_.model_3d_path);
    printf("3D model loaded.\n");

    if (use_gpu_) {
        net_2d_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_2d_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        net_3d_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_3d_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }

    printf("Loading NuScenes tables from: %s ...\n", config_.nuscenes_root.c_str());
    tables_ = nuscenes::LoadTables(config_.nuscenes_root);
    printf("Tables loaded.\n");

    // GPU warmup: first forward compiles CUDA kernels with correct input shapes.
    if (use_gpu_) {
        printf("GPU warmup...\n");
        auto t_warm = Clock::now();

        int w2d[4] = {24, 3, 256, 704};
        cv::Mat dummy_2d(4, w2d, CV_32F, cv::Scalar(0));
        net_2d_.setInput(dummy_2d);
        net_2d_.forward();

        int w3d[5] = {1, 200, 200, 4, 256};
        cv::Mat dummy_3d(5, w3d, CV_32F, cv::Scalar(0));
        net_3d_.setInput(dummy_3d);
        auto out_names = net_3d_.getUnconnectedOutLayersNames();
        std::vector<cv::Mat> tmp;
        net_3d_.forward(tmp, out_names);

        printf("GPU warmup done (%.0f ms)\n", Elapsed(t_warm));
    }

    printf("FastBEVInfer initialized.\n");
    printf("  GPU: %s\n", use_gpu_ ? "enabled" : "disabled");
}
// ==================================================================
// BuildInputBlob — HWC image tensors → NCHW blob
// ==================================================================

cv::Mat FastBEVInfer::BuildInputBlob(const std::string& image_dir,
                                      const nuscenes::AugParams* aug) const
{
    int N = nuscenes::N_SWEEPS * 6;
    int H = nuscenes::IMG_SIZE.height;
    int W = nuscenes::IMG_SIZE.width;
    int blob_dims[4] = {N, 3, H, W};
    cv::Mat blob(4, blob_dims, CV_32F);

    std::atomic<bool> has_error{false};
    std::string error_msg;

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < N; ++idx) {
        if (has_error.load(std::memory_order_relaxed)) continue;

        try {
            int s = idx / 6;
            int c = idx % 6;
            const auto& cam = nuscenes::CAMERA_ORDER[c];

            // Find image file
            std::string path;
            for (const auto& ext : {".png", ".jpg", ".jpeg"}) {
                auto p = fs::path(image_dir) / (cam + "_sweep" + std::to_string(s) + ext);
                if (fs::exists(p)) { path = p.string(); break; }
            }
            if (path.empty())
                throw std::runtime_error("Missing " + cam + "_sweep" + std::to_string(s));

            cv::Mat bgr = cv::imread(path);
            if (bgr.empty())
                throw std::runtime_error("Failed to read " + path);

            // Resize + optional crop → contiguous uint8
            cv::Mat resized_temp, img_u8;
            if (aug) {
                cv::resize(bgr, resized_temp, aug->resize_dims, 0, 0, cv::INTER_LINEAR);
                img_u8 = resized_temp(aug->crop).clone();
            } else {
                cv::resize(bgr, resized_temp, nuscenes::IMG_SIZE, 0, 0, cv::INTER_LINEAR);
                img_u8 = resized_temp;
            }

            // Split uint8 BGR → normalize with original (x-mean)/std → directly into blob
            std::vector<cv::Mat> chs(3);
            cv::split(img_u8, chs);
            cv::Mat dst_r(H, W, CV_32F, blob.ptr(idx, 0));
            cv::Mat dst_g(H, W, CV_32F, blob.ptr(idx, 1));
            cv::Mat dst_b(H, W, CV_32F, blob.ptr(idx, 2));
            chs[2].convertTo(dst_r, CV_32F);  cv::subtract(dst_r, IMG_MEAN[0], dst_r);  cv::divide(dst_r, IMG_STD[0], dst_r);
            chs[1].convertTo(dst_g, CV_32F);  cv::subtract(dst_g, IMG_MEAN[1], dst_g);  cv::divide(dst_g, IMG_STD[1], dst_g);
            chs[0].convertTo(dst_b, CV_32F);  cv::subtract(dst_b, IMG_MEAN[2], dst_b);  cv::divide(dst_b, IMG_STD[2], dst_b);
        } catch (const std::exception& e) {
            has_error.store(true, std::memory_order_relaxed);
            error_msg = e.what();
        }
    }

    if (has_error.load(std::memory_order_relaxed))
        throw std::runtime_error(error_msg);

    return blob;
}

// ==================================================================
// FeatureToBEV — dispatches to bev_pool (OpenMP / CUDA)
// ==================================================================

cv::Mat FastBEVInfer::FeatureToBEV(const cv::Mat& feat_2d,
                                    const std::vector<cv::Mat>& extrinsics) const
{
    return bev_pool::FeatureToBEV(feat_2d, extrinsics,
                                   nuscenes::N_SWEEPS, 6,
                                   nuscenes::IMG_SIZE.width,
                                   use_gpu_);
}

// ==================================================================
// PostProcess — decode + NMS → detections
// ==================================================================

std::vector<visualize::Detection> FastBEVInfer::PostProcess(
    const cv::Mat& cls_out, const cv::Mat& bbox_out, const cv::Mat& dir_out) const
{
    int feat_h = cls_out.size[2];
    int feat_w = cls_out.size[3];
    int total_anchors = feat_h * feat_w * 8;  // 80000
    int n_per_pos = 8;

    // ---- Reshape cls: [1, 80, 100, 100] → [80000, 10] ----
    cv::Mat cls_t(total_anchors, postprocess::NUM_CLASSES, CV_32F);
    for (int h = 0; h < feat_h; ++h) {
        for (int w = 0; w < feat_w; ++w) {
            for (int a = 0; a < n_per_pos; ++a) {
                int i = (h * feat_w + w) * n_per_pos + a;
                int base_ch = a * postprocess::NUM_CLASSES;
                for (int c = 0; c < postprocess::NUM_CLASSES; ++c)
                    cls_t.at<float>(i, c) = cls_out.ptr<float>(0, base_ch + c)[h * feat_w + w];
            }
        }
    }

    // ---- Reshape bbox: [1, 72, 100, 100] → [80000, 9] ----
    cv::Mat bbox_t(total_anchors, postprocess::BOX_CODE_SIZE, CV_32F);
    for (int h = 0; h < feat_h; ++h) {
        for (int w = 0; w < feat_w; ++w) {
            for (int a = 0; a < n_per_pos; ++a) {
                int i = (h * feat_w + w) * n_per_pos + a;
                int base_ch = a * postprocess::BOX_CODE_SIZE;
                for (int c = 0; c < postprocess::BOX_CODE_SIZE; ++c)
                    bbox_t.at<float>(i, c) = bbox_out.ptr<float>(0, base_ch + c)[h * feat_w + w];
            }
        }
    }

    // ---- Reshape dir: [1, 16, 100, 100] → [80000, 2] ----
    cv::Mat dir_t(total_anchors, 2, CV_32F);
    for (int h = 0; h < feat_h; ++h) {
        for (int w = 0; w < feat_w; ++w) {
            for (int a = 0; a < n_per_pos; ++a) {
                int i = (h * feat_w + w) * n_per_pos + a;
                int base_ch = a * 2;
                for (int c = 0; c < 2; ++c)
                    dir_t.at<float>(i, c) = dir_out.ptr<float>(0, base_ch + c)[h * feat_w + w];
            }
        }
    }

    // ---- Anchors ----
    auto anchors = postprocess::GenerateAnchors(feat_h, feat_w);

    // ---- NMS_PRE: top-K by max class score ----
    if (total_anchors > postprocess::NMS_PRE) {
        std::vector<std::pair<float, int>> max_scores;
        max_scores.reserve(total_anchors);
        for (int i = 0; i < total_anchors; ++i) {
            float ms = 0;
            for (int c = 0; c < postprocess::NUM_CLASSES; ++c)
                ms = std::max(ms, cls_t.at<float>(i, c));
            max_scores.push_back({ms, i});
        }
        std::partial_sort(max_scores.begin(), max_scores.begin() + postprocess::NMS_PRE,
                          max_scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });

        cv::Mat cls_pre(postprocess::NMS_PRE, postprocess::NUM_CLASSES, CV_32F);
        cv::Mat bbox_pre(postprocess::NMS_PRE, postprocess::BOX_CODE_SIZE, CV_32F);
        cv::Mat dir_pre(postprocess::NMS_PRE, 2, CV_32F);
        cv::Mat anchors_pre(postprocess::NMS_PRE, 9, CV_32F);

        for (int i = 0; i < postprocess::NMS_PRE; ++i) {
            int idx = max_scores[i].second;
            anchors.row(idx).copyTo(anchors_pre.row(i));
            cls_t.row(idx).copyTo(cls_pre.row(i));
            bbox_t.row(idx).copyTo(bbox_pre.row(i));
            dir_t.row(idx).copyTo(dir_pre.row(i));
        }
        cls_t = cls_pre;
        bbox_t = bbox_pre;
        dir_t = dir_pre;
        anchors = anchors_pre;
    }

    // ---- Direction scores ----
    std::vector<int> dir_scores;
    dir_scores.reserve(dir_t.rows);
    for (int i = 0; i < dir_t.rows; ++i)
        dir_scores.push_back(dir_t.at<float>(i, 0) > dir_t.at<float>(i, 1) ? 0 : 1);

    // ---- Decode + NMS ----
    auto decoded_boxes = postprocess::DecodeBoxes(anchors, bbox_t);
    auto nms_result = postprocess::MultiClassScaleNMS(decoded_boxes, cls_t, dir_scores);

    if (nms_result.boxes.rows > 0)
        postprocess::CorrectYaw(nms_result.boxes, nms_result.dir_scores);

    // ---- Build Detection list ----
    std::vector<visualize::Detection> detections;
    for (int i = 0; i < nms_result.boxes.rows; ++i) {
        if (nms_result.scores[i] < config_.score_thr) continue;
        const float* b = nms_result.boxes.ptr<float>(i);
        visualize::Detection det;
        det.x = b[0]; det.y = b[1]; det.z = b[2];
        det.w = b[3]; det.l = b[4]; det.h = b[5]; det.yaw = b[6];
        det.score = nms_result.scores[i];
        det.label = nms_result.labels[i];
        detections.push_back(det);
    }

    std::sort(detections.begin(), detections.end(), [](auto& a, auto& b) {
        return a.score > b.score;
    });

    return detections;
}

// ==================================================================
// Run — main inference pipeline
// ==================================================================

std::vector<visualize::Detection> FastBEVInfer::Run(const std::string& sample_token)
{
    auto t_total = Clock::now();

    printf("\n=== FastBEV C++ Inference ===\n");
    printf("Sample token: %s\n", sample_token.c_str());

    // ---- Step 1: Data preparation ----
    auto t_prep = Clock::now();

    auto temporal_tokens = nuscenes::GetTemporalSampleTokens(
        sample_token, nuscenes::N_SWEEPS, tables_);
    printf("Temporal keyframes:\n");
    for (size_t i = 0; i < temporal_tokens.size(); ++i)
        printf("  sweep%zu: %s\n", i, temporal_tokens[i].c_str());

    if (!config_.no_export) {
        nuscenes::ExportTemporalImages(temporal_tokens, config_.image_dir,
                                        config_.dataroot, tables_);
        printf("Exported %dx6 images -> %s\n", nuscenes::N_SWEEPS, config_.image_dir.c_str());
    }
    last_image_dir_ = config_.image_dir;

    nuscenes::AugParams aug;
    auto extrinsics = nuscenes::BuildTemporalExtrinsics(
        temporal_tokens, nuscenes::ORIG_IMG_SIZE.height,
        nuscenes::ORIG_IMG_SIZE.width, tables_, &aug);

    last_lidar2img_native_ = nuscenes::LoadSampleCalibNative(temporal_tokens[0], tables_);

    cv::Mat img_blob = BuildInputBlob(config_.image_dir, &aug);

    double ms_prep = Elapsed(t_prep);

    // ---- Step 2: 2D model inference ----
    printf("Running 2D model...\n");
    auto t_2d_set = Clock::now();
    net_2d_.setInput(img_blob);
    double ms_2d_set = Elapsed(t_2d_set);

    auto t_2d_fwd = Clock::now();
    auto feat_2d_out = net_2d_.forward();
    double ms_2d_fwd = Elapsed(t_2d_fwd);
    double ms_2d = ms_2d_set + ms_2d_fwd;

    printf("2D features: [%d, %d, %d, %d]  (setInput=%.1f ms, forward=%.1f ms)\n",
           feat_2d_out.size[0], feat_2d_out.size[1],
           feat_2d_out.size[2], feat_2d_out.size[3],
           ms_2d_set, ms_2d_fwd);

    // ---- Step 3: Feature → BEV ----
    printf("Backprojecting to BEV...\n");
    auto t_bev = Clock::now();
    cv::Mat volume_input = FeatureToBEV(feat_2d_out, extrinsics);
    double ms_bev = Elapsed(t_bev);

    // ---- Step 4: 3D model inference ----
    printf("Running 3D model...\n");
    auto out_names = net_3d_.getUnconnectedOutLayersNames();

    auto t_3d_set = Clock::now();
    net_3d_.setInput(volume_input);
    double ms_3d_set = Elapsed(t_3d_set);

    auto t_3d_fwd = Clock::now();
    std::vector<cv::Mat> out_3d;
    net_3d_.forward(out_3d, out_names);
    double ms_3d_fwd = Elapsed(t_3d_fwd);
    double ms_3d = ms_3d_set + ms_3d_fwd;

    if (out_3d.size() < 3)
        throw std::runtime_error("3D model expected 3 outputs, got " + std::to_string(out_3d.size()));

    // Identify outputs by channel count: cls=80, bbox=72, dir=16
    cv::Mat cls_out, bbox_out, dir_out;
    for (const auto& m : out_3d) {
        int ch = m.size[1];
        if (ch == 80)       cls_out = m;
        else if (ch == 72)  bbox_out = m;
        else if (ch == 16)  dir_out = m;
    }
    if (cls_out.empty() || bbox_out.empty() || dir_out.empty())
        throw std::runtime_error("Failed to identify 3D outputs by channel count");

    printf("3D outputs: cls [%d,%d,%d,%d]  bbox [%d,%d,%d,%d]  dir [%d,%d,%d,%d]\n",
           cls_out.size[0], cls_out.size[1], cls_out.size[2], cls_out.size[3],
           bbox_out.size[0], bbox_out.size[1], bbox_out.size[2], bbox_out.size[3],
           dir_out.size[0], dir_out.size[1], dir_out.size[2], dir_out.size[3]);

    // ---- Step 5: Post-processing ----
    printf("Post-processing (decode + NMS)...\n");
    auto t_post = Clock::now();
    auto detections = PostProcess(cls_out, bbox_out, dir_out);
    double ms_post = Elapsed(t_post);

    double ms_total = Elapsed(t_total);

    // ---- Timing summary ----
    printf("\n");
    printf("  Data prep:       %8.1f ms\n", ms_prep);
    printf("  2D model:        %8.1f ms  (setInput=%.1f, forward=%.1f)\n", ms_2d, ms_2d_set, ms_2d_fwd);
    printf("  Feature -> BEV:  %8.1f ms\n", ms_bev);
    printf("  3D model:        %8.1f ms  (setInput=%.1f, forward=%.1f)\n", ms_3d, ms_3d_set, ms_3d_fwd);
    printf("  Post-process:    %8.1f ms\n", ms_post);
    printf("  ----------------------------\n");
    printf("  Total:           %8.1f ms\n", ms_total);

    // ---- Print top detections ----
    int n_show = std::min(15, (int)detections.size());
    printf("\nDetections: %zu (score >= %.2f)\n", detections.size(), config_.score_thr);
    for (int i = 0; i < n_show; ++i) {
        auto& d = detections[i];
        printf("  %-20s %.3f  center=(%.1f,%.1f,%.1f)  size=(%.1f,%.1f,%.1f)\n",
               visualize::CATEGORIES[d.label].c_str(), d.score,
               d.x, d.y, d.z, d.w, d.l, d.h);
    }

    return detections;
}

// ==================================================================
// Visualization
// ==================================================================

void FastBEVInfer::Visualize(const std::vector<visualize::Detection>& detections,
                              const std::string& image_dir_override) const
{
    std::string img_dir = image_dir_override.empty() ? last_image_dir_ : image_dir_override;
    visualize::VisualizeResults(detections, img_dir, config_.out_dir,
                                last_lidar2img_native_,
                                nuscenes::CAMERA_ORDER, 0.15f);
    printf("\nDone. Outputs: %s\n", config_.out_dir.c_str());
}
