#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

namespace nuscenes {

// ---- Constants ----
const std::vector<std::string> CAMERA_ORDER = {
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
};
const int N_SWEEPS = 4;
const cv::Size ORIG_IMG_SIZE(1600, 900);
const cv::Size IMG_SIZE(704, 256);

struct ImageAugCfg {
    cv::Size src_size = cv::Size(1600, 900);
    cv::Size test_input_size = cv::Size(704, 256);
    float test_resize = 0.0f;
    float test_rotate = 0.0f;
    bool  test_flip = false;
};

struct AugParams {
    float resize;
    cv::Size resize_dims;
    cv::Rect crop;
    bool flip;
    float rotate;
    int fH, fW;
};

using json = nlohmann::json;

// Tables holds loaded NuScenes JSON data.
struct Tables {
    std::unordered_map<std::string, json> calibrated_sensor;
    std::unordered_map<std::string, json> ego_pose;
    std::vector<json> sample_data_list;
    std::vector<json> sample_list;
    std::unordered_map<std::string, json> sample_by_token;
};

// ---- Table loading ----
Tables LoadTables(const std::string& nuscenes_root);

// ---- Temporal tokens ----
std::vector<std::string> GetTemporalSampleTokens(
    const std::string& current_token, int n_times, const Tables& tables);

// ---- Image paths ----
std::unordered_map<std::string, std::string> GetSampleImagePaths(
    const std::string& sample_token, const Tables& tables);

// ---- Export images ----
void ExportTemporalImages(
    const std::vector<std::string>& temporal_tokens,
    const std::string& image_dir,
    const std::string& dataroot,
    const Tables& tables);

// ---- Calibration ----
// Returns lidar2img (3x4) per camera (native, no augmentation)
std::unordered_map<std::string, cv::Mat> LoadSampleCalibNative(
    const std::string& sample_token, const Tables& tables);

// Returns {intrin(3x3), rot(3x3), tran(3x1)} per camera
struct CamCalibAug {
    cv::Mat intrin;  // 3x3
    cv::Mat rot;     // 3x3
    cv::Mat tran;    // 3x1
};
std::unordered_map<std::string, CamCalibAug> LoadSampleCalibAug(
    const std::string& sample_token,
    const std::string& ref_sample_token,
    const Tables& tables);

// ---- Augmentation ----
AugParams SampleTestAugmentation(int height, int width,
                                  const ImageAugCfg& cfg = ImageAugCfg());

// Returns post_rot (3x3), post_tran (3x1)
std::pair<cv::Mat, cv::Mat> ImageTransformPostMatrix(const AugParams& aug);

// ---- Projection ----
// Returns 4x4 lidar2img matrix
cv::Mat Rts2Proj(const CamCalibAug& cam_info,
                 const cv::Mat& post_rot, const cv::Mat& post_tran);

// Build extrinsics for all temporal frames
std::vector<cv::Mat> BuildTemporalExtrinsics(
    const std::vector<std::string>& temporal_tokens,
    int height, int width,
    const Tables& tables,
    AugParams* out_aug = nullptr,
    const ImageAugCfg& data_cfg = ImageAugCfg());

}  // namespace nuscenes
