#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>

namespace visualize {

struct Detection {
    float x, y, z, w, l, h, yaw;
    float score;
    int label;
};

extern const std::vector<std::string> CATEGORIES;
extern const std::vector<cv::Scalar> COLORS;

// Draw BEV top-down view, return as 1000x1000 BGR image.
cv::Mat DrawBEVFrame(const std::vector<Detection>& dets, float score_thr = 0.15f);

// Combined frame for video: camera images (3x2) with 3D boxes on left,
// BEV top-down view on right. Canvas: 2000 x 1000.
cv::Mat DrawCombinedFrame(
    const std::vector<Detection>& dets,
    const std::vector<cv::Mat>& cam_images,
    const std::vector<std::string>& cam_names,
    const std::unordered_map<std::string, cv::Mat>& lidar2img_native,
    float score_thr = 0.15f);

// Draw all visualizations and save to out_dir.
// - BEV top-down view (bev_detections.png)
// - Camera input images grid (camera_inputs.png)
// - Camera images with 3D box projections (camera_3d_boxes.png)
void VisualizeResults(
    const std::vector<Detection>& detections,
    const std::string& image_dir,
    const std::string& out_dir,
    const std::unordered_map<std::string, cv::Mat>& lidar2img_native,
    const std::vector<std::string>& camera_order,
    float score_thr = 0.15f);

}  // namespace visualize
