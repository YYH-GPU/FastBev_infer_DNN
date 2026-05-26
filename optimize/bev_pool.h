#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace bev_pool {

// BEV grid constants
constexpr int VX = 200, VY = 200, VZ = 4;
constexpr float VS_X = 0.5f, VS_Y = 0.5f, VS_Z = 1.5f;
constexpr float PC_MIN[3] = {-50.f, -50.f, -5.f};
constexpr float PC_MAX[3] = { 50.f,  50.f,  3.f};

// Optimized Feature → BEV volume pipeline.
// Dispatches to CUDA or OpenMP depending on use_gpu and compile-time flags.
//
// feat_2d:      [N, H, W, C]  NHWC from 2D model  (N=24, H=64, W=176, C=64)
// extrinsics:   N projection matrices [3, 4]  (lidar2img, not yet stride-scaled)
// n_sweeps:     temporal sweeps (4)
// n_cams:       cameras per sweep (6)
// feat_w:       image feature width (176)
// use_gpu:      true → CUDA kernel, false → OpenMP
//
// NHWC → NCHW transpose with OpenMP. Shared by CPU and CUDA paths.
void TransposeNHWC2NCHW(const float* src, float* dst,
                         int N, int H, int W, int C);

// Returns: [1, 200, 200, 4, 256]  ready for 3D model input.
cv::Mat FeatureToBEV(const cv::Mat& feat_2d,
                      const std::vector<cv::Mat>& extrinsics,
                      int n_sweeps, int n_cams, int feat_w,
                      bool use_gpu);

}  // namespace bev_pool
