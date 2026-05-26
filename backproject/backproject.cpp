#include "backproject.h"
#include <cmath>
#include <cstring>

namespace backproject {

cv::Mat GetPoints(const int n_voxels[3],
                  const float voxel_size[3],
                  const float origin[3])
{
    int vx = n_voxels[0], vy = n_voxels[1], vz = n_voxels[2];

    float new_origin[3];
    for (int i = 0; i < 3; ++i)
        new_origin[i] = origin[i] - (n_voxels[i] / 2.0f) * voxel_size[i];

    // points: [vx, vy, vz, 3] — last dim is (px, py, pz)
    int dims[4] = {vx, vy, vz, 3};
    cv::Mat points(4, dims, CV_32F);

    for (int z = 0; z < vz; ++z) {
        for (int y = 0; y < vy; ++y) {
            for (int x = 0; x < vx; ++x) {
                float* ptr = reinterpret_cast<float*>(points.ptr(x, y, z));
                ptr[0] = x * voxel_size[0] + new_origin[0];
                ptr[1] = y * voxel_size[1] + new_origin[1];
                ptr[2] = z * voxel_size[2] + new_origin[2];
            }
        }
    }
    return points;
}

cv::Mat BackprojectInplace(const cv::Mat& features,
                           const cv::Mat& points,
                           const cv::Mat& projection)
{
    // features: [N, C, H, W] (NCHW)
    int n_images = features.size[0];
    int C = features.size[1];
    int H = features.size[2];
    int W = features.size[3];

    // points: [vx, vy, vz, 3]
    int vx = points.size[0];
    int vy = points.size[1];
    int vz = points.size[2];

    // projection: [N, 3, 4]
    CV_Assert(projection.size[0] == n_images);

    // volume: [C, vx, vy, vz]
    int vol_dims[4] = {C, vx, vy, vz};
    cv::Mat volume(4, vol_dims, CV_32F, cv::Scalar(0));

    int vol_ystride = vz;  // last dim stride for [vy, vz] slice

    for (int i = 0; i < n_images; ++i) {
        for (int z = 0; z < vz; ++z) {
            for (int y = 0; y < vy; ++y) {
                for (int x = 0; x < vx; ++x) {
                    const float* pt = reinterpret_cast<const float*>(points.ptr(x, y, z));
                    float px = pt[0], py = pt[1], pz = pt[2];

                    float u = projection.at<float>(i, 0, 0) * px +
                              projection.at<float>(i, 0, 1) * py +
                              projection.at<float>(i, 0, 2) * pz +
                              projection.at<float>(i, 0, 3);
                    float v = projection.at<float>(i, 1, 0) * px +
                              projection.at<float>(i, 1, 1) * py +
                              projection.at<float>(i, 1, 2) * pz +
                              projection.at<float>(i, 1, 3);
                    float depth = projection.at<float>(i, 2, 0) * px +
                                  projection.at<float>(i, 2, 1) * py +
                                  projection.at<float>(i, 2, 2) * pz +
                                  projection.at<float>(i, 2, 3);

                    int ix = static_cast<int>(std::round(u / depth));
                    int iy = static_cast<int>(std::round(v / depth));

                    if (ix < 0 || iy < 0 || ix >= W || iy >= H || depth <= 0)
                        continue;

                    // Write features to volume: volume[c, x, y, z] = features[i, c, iy, ix]
                    for (int c = 0; c < C; ++c) {
                        volume.ptr<float>(c, x)[y * vol_ystride + z] =
                            features.ptr<float>(i, c)[iy * W + ix];
                    }
                }
            }
        }
    }
    return volume;
}

}  // namespace backproject
