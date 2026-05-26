#include "bev_pool.h"
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WITH_CUDA
namespace bev_pool {
// Defined in bev_pool_cuda.cu
cv::Mat FeatureToBEV_CUDA(const cv::Mat&, const std::vector<cv::Mat>&,
                           int, int, int);
}
#endif

namespace bev_pool {

namespace {

// Precomputed voxel center coordinates
struct GridPoints {
    float px[VX], py[VY], pz[VZ];

    GridPoints() {
        float ox = (PC_MIN[0] + PC_MAX[0]) * 0.5f - VX * VS_X * 0.5f;
        float oy = (PC_MIN[1] + PC_MAX[1]) * 0.5f - VY * VS_Y * 0.5f;
        float oz = (PC_MIN[2] + PC_MAX[2]) * 0.5f - VZ * VS_Z * 0.5f;
        for (int x = 0; x < VX; ++x) px[x] = x * VS_X + ox;
        for (int y = 0; y < VY; ++y) py[y] = y * VS_Y + oy;
        for (int z = 0; z < VZ; ++z) pz[z] = z * VS_Z + oz;
    }
};

static const GridPoints g_pts;

// OpenMP-accelerated backprojection
cv::Mat BackprojectOMP(const float* features, const float* proj,
                        int N, int C, int H, int W)
{
    int cstride_xy = VY * VZ;
    int vol_size = C * VX * VY * VZ;
    int vol_cstride = VX * VY * VZ;

    cv::Mat volume(1, vol_size, CV_32F);
    float* vol = reinterpret_cast<float*>(volume.data);
    std::memset(vol, 0, vol_size * sizeof(float));

    for (int i = 0; i < N; ++i) {
        const float* p = proj + i * 12;
        const float* feat_i = features + i * C * H * W;

        float p00 = p[0], p01 = p[1], p02 = p[2],  p03 = p[3];
        float p10 = p[4], p11 = p[5], p12 = p[6],  p13 = p[7];
        float p20 = p[8], p21 = p[9], p22 = p[10], p23 = p[11];

        #pragma omp parallel for collapse(3)
        for (int z = 0; z < VZ; ++z) {
            for (int y = 0; y < VY; ++y) {
                for (int x = 0; x < VX; ++x) {
                    float px = g_pts.px[x];
                    float py = g_pts.py[y];
                    float pz = g_pts.pz[z];

                    float u = p00 * px + p01 * py + p02 * pz + p03;
                    float v = p10 * px + p11 * py + p12 * pz + p13;
                    float depth = p20 * px + p21 * py + p22 * pz + p23;

                    if (depth <= 0) continue;

                    int ix = static_cast<int>(std::round(u / depth));
                    int iy = static_cast<int>(std::round(v / depth));

                    if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;

                    int feat_off = iy * W + ix;
                    int vol_off = cstride_xy * x + VZ * y + z;

                    for (int c = 0; c < C; ++c)
                        vol[c * vol_cstride + vol_off] = feat_i[c * H * W + feat_off];
                }
            }
        }
    }
    return volume;
}

}  // anonymous namespace

// NHWC → NCHW transpose with OpenMP (exposed for CUDA path reuse)
void TransposeNHWC2NCHW(const float* src, float* dst,
                         int N, int H, int W, int C)
{
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    dst[n * C * H * W + c * H * W + h * W + w] =
                        src[n * H * W * C + h * W * C + w * C + c];
                }
            }
        }
    }
}

// ==================================================================
// OpenMP implementation
// ==================================================================

static cv::Mat FeatureToBEV_OMP(const cv::Mat& feat_2d,
                                 const std::vector<cv::Mat>& extrinsics,
                                 int n_sweeps, int n_cams, int feat_w)
{
    using Clock = std::chrono::high_resolution_clock;
    double t_trans = 0, t_proj = 0, t_back = 0, t_scat = 0;

    int H_feat = feat_2d.size[1];
    int W_feat = feat_2d.size[2];
    int C_feat = feat_2d.size[3];
    int C_total = C_feat * n_sweeps;

    float stride_f = static_cast<float>(feat_w) / W_feat;
    const float* feat_nhwc = reinterpret_cast<const float*>(feat_2d.data);

    int out_elems = VX * VY * VZ * C_total;
    cv::Mat result(1, out_elems, CV_32F);
    float* out = reinterpret_cast<float*>(result.data);

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {

        // 1. NHWC → NCHW
        auto t0 = Clock::now();
        cv::Mat nchw(1, n_cams * C_feat * H_feat * W_feat, CV_32F);
        float* nchw_ptr = reinterpret_cast<float*>(nchw.data);
        TransposeNHWC2NCHW(feat_nhwc + sweep * n_cams * H_feat * W_feat * C_feat,
                            nchw_ptr, n_cams, H_feat, W_feat, C_feat);
        t_trans += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // 2. Flat projection [n_cams, 12]
        t0 = Clock::now();
        cv::Mat proj_flat(1, n_cams * 12, CV_32F);
        float* proj = reinterpret_cast<float*>(proj_flat.data);
        for (int i = 0; i < n_cams; ++i) {
            const float* ext = reinterpret_cast<const float*>(
                extrinsics[sweep * n_cams + i].data);
            float* dst = proj + i * 12;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    float val = ext[r * 4 + c];
                    if (r < 2) val /= stride_f;
                    dst[r * 4 + c] = val;
                }
            }
        }
        t_proj += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // 3. Backprojection
        t0 = Clock::now();
        cv::Mat vol = BackprojectOMP(nchw_ptr, proj, n_cams, C_feat, H_feat, W_feat);
        t_back += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        const float* vol_ptr = reinterpret_cast<const float*>(vol.data);

        // 4. Scatter into output [C_total, VX, VY, VZ]
        t0 = Clock::now();
        #pragma omp parallel for collapse(2)
        for (int x = 0; x < VX; ++x) {
            for (int y = 0; y < VY; ++y) {
                for (int z = 0; z < VZ; ++z) {
                    int dst_base = x * VY * VZ * C_total + y * VZ * C_total + z * C_total;
                    int src_base = x * VY * VZ + y * VZ + z;
                    for (int c = 0; c < C_feat; ++c) {
                        int dst_c = sweep * C_feat + c;
                        out[dst_base + dst_c] = vol_ptr[c * VX * VY * VZ + src_base];
                    }
                }
            }
        }
        t_scat += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    int dims[5] = {1, VX, VY, VZ, C_total};
    auto result_5d = cv::Mat(5, dims, CV_32F, out).clone();
    printf("  [OMP] transpose=%.1f proj=%.1f backproject=%.1f scatter=%.1f ms\n",
           t_trans, t_proj, t_back, t_scat);
    return result_5d;
}

// ==================================================================
// Dispatcher
// ==================================================================

cv::Mat FeatureToBEV(const cv::Mat& feat_2d,
                      const std::vector<cv::Mat>& extrinsics,
                      int n_sweeps, int n_cams, int feat_w,
                      bool use_gpu)
{
#ifdef WITH_CUDA
    if (use_gpu)
        return FeatureToBEV_CUDA(feat_2d, extrinsics, n_sweeps, n_cams, feat_w);
#endif
    return FeatureToBEV_OMP(feat_2d, extrinsics, n_sweeps, n_cams, feat_w);
}

}  // namespace bev_pool
