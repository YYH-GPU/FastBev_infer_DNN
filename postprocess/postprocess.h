#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace postprocess {

// ---- Constants ----
constexpr int NUM_CLASSES = 10;
constexpr int BOX_CODE_SIZE = 9;

extern const float ANCHOR_SIZES[4][3];    // 4 size groups x (w,l,h)
extern const float ANCHOR_ROTATIONS[2];   // 0, 1.57
extern const float ANCHOR_RANGE[6];       // [xmin,ymin,zmin,xmax,ymax,zmax]
extern const int NMS_PRE;                 // 200
extern const int MAX_NUM;                 // 50
extern const float NMS_SCORE_THR;         // 0.3
extern const std::vector<std::string> NMS_TYPE_LIST;
extern const float NMS_THR_LIST[10];
extern const float NMS_RADIUS_THR_LIST[10];
extern const float NMS_RESCALE_FACTOR[10];
extern const float DIR_OFFSET;

// ---- Anchor generation ----
// Returns [num_anchors, 9]  (x,y,z,w,l,h,r,vx,vy) with vx=vy=0
cv::Mat GenerateAnchors(int feat_h, int feat_w);

// ---- Box decoding ----
// anchors: [N, 9], deltas: [N, 9]
// Returns: [N, 9] decoded boxes
cv::Mat DecodeBoxes(const cv::Mat& anchors, const cv::Mat& deltas);

// ---- NMS ----
std::vector<int> CircleNMS(const cv::Mat& dets, float thresh);

std::vector<int> RotateNMSCPU(const cv::Mat& boxes_bev,
                              const std::vector<float>& scores, float thresh);

// Per-class scale-NMS
// all_boxes: [N, 9], all_scores: [N, NUM_CLASSES], all_dir_scores: [N]
// Returns: {boxes [M,9], scores [M], labels [M], dir_scores [M]}
struct NMSResult {
    cv::Mat boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> dir_scores;
};
NMSResult MultiClassScaleNMS(const cv::Mat& all_boxes,
                              const cv::Mat& all_scores,
                              const std::vector<int>& all_dir_scores);

// Direction correction
void CorrectYaw(cv::Mat& boxes, const std::vector<int>& dir_scores);

}  // namespace postprocess
