#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace backproject {

// Generate 3D grid points in LiDAR coordinates.
// n_voxels: [X, Y, Z] grid counts
// voxel_size: [X, Y, Z] physical size per voxel (meters)
// origin: [3] center of the grid in LiDAR coordinates
// Returns: [vx, vy, vz, 3] float32  (last dim = px, py, pz)
cv::Mat GetPoints(const int n_voxels[3],
                  const float voxel_size[3],
                  const float origin[3]);

// Backproject 2D image features into 3D BEV volume.
// features: [N, C, H, W]  (NCHW)
// points: [vx, vy, vz, 3]  grid points from GetPoints()
// projection: [N, 3, 4] lidar2img matrices (already stride-scaled)
// Returns: [C, vx, vy, vz]
cv::Mat BackprojectInplace(const cv::Mat& features,
                           const cv::Mat& points,
                           const cv::Mat& projection);

}  // namespace backproject
