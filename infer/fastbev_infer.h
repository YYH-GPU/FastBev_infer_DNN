#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "nuscenes_loader.h"
#include "visualize.h"

class FastBEVInfer {
public:
    struct Config {
        std::string model_2d_path;
        std::string model_3d_path;
        std::string nuscenes_root;
        std::string dataroot;
        std::string image_dir;
        std::string out_dir;
        float score_thr = 0.3f;
        bool use_gpu = true;
        bool no_export = false;
    };

    explicit FastBEVInfer(const Config& config);
    ~FastBEVInfer() = default;

    // Run inference on a single NuScenes sample token.
    std::vector<visualize::Detection> Run(const std::string& sample_token);

    void Visualize(const std::vector<visualize::Detection>& detections,
                   const std::string& image_dir_override = "") const;

    const std::unordered_map<std::string, cv::Mat>& GetLastCalib() const {
        return last_lidar2img_native_;
    }

    const std::string& GetLastImageDir() const {
        return last_image_dir_;
    }

private:
    // ---- Pipeline stages ----

    // Load & preprocess images, stack into NCHW blob [N, C, H, W].
    cv::Mat BuildInputBlob(const std::string& image_dir,
                           const nuscenes::AugParams* aug) const;

    // Backproject 2D features into BEV volume for 3D model input.
    // feat_2d: [N*C, H, W, C_feat] NHWC from 2D model
    // extrinsics: N*C projection matrices [3, 4]
    // Returns: [1, 200, 200, 4, 256]
    cv::Mat FeatureToBEV(const cv::Mat& feat_2d,
                          const std::vector<cv::Mat>& extrinsics) const;

    // Decode + NMS → detections.
    std::vector<visualize::Detection> PostProcess(
        const cv::Mat& cls_out, const cv::Mat& bbox_out, const cv::Mat& dir_out) const;

    // ---- State ----

    Config config_;
    bool use_gpu_ = false;  // effective GPU mode after runtime detection
    cv::dnn::Net net_2d_;
    cv::dnn::Net net_3d_;
    nuscenes::Tables tables_;

    std::unordered_map<std::string, cv::Mat> last_lidar2img_native_;
    std::string last_image_dir_;
};
