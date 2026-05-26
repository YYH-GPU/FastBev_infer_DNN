#include "bev_pool.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bev_pool {

namespace {

// Precomputed grid points (duplicated from bev_pool.cpp for standalone .cu compilation)
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

// ------------------------------------------------------------------
// CUDA kernel: one thread per (image, x, y), loops over z and c
// ------------------------------------------------------------------
__global__ void BackprojectKernel(
    const float* features, const float* proj, float* volume,
    const float* px_arr, const float* py_arr, const float* pz_arr,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int imgs_xy = VX * VY;

    int i = idx / imgs_xy;
    int r = idx - i * imgs_xy;
    int x = r / VY;
    int y = r % VY;

    if (i >= N || x >= VX || y >= VY) return;

    const float* proj_i = proj + i * 12;
    float p00 = proj_i[0], p01 = proj_i[1], p02 = proj_i[2],  p03 = proj_i[3];
    float p10 = proj_i[4], p11 = proj_i[5], p12 = proj_i[6],  p13 = proj_i[7];
    float p20 = proj_i[8], p21 = proj_i[9], p22 = proj_i[10], p23 = proj_i[11];

    const float* feat_i = features + i * C * H * W;
    int vol_cstride = VX * VY * VZ;
    int cstride_xy = VY * VZ;

    float vx = px_arr[x];
    float vy = py_arr[y];

    for (int z = 0; z < VZ; ++z) {
        float vz = pz_arr[z];

        float u = p00 * vx + p01 * vy + p02 * vz + p03;
        float v = p10 * vx + p11 * vy + p12 * vz + p13;
        float depth = p20 * vx + p21 * vy + p22 * vz + p23;

        if (depth <= 0) continue;

        int ix = static_cast<int>(roundf(u / depth));
        int iy = static_cast<int>(roundf(v / depth));

        if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;

        int feat_off = iy * W + ix;
        int vol_off = cstride_xy * x + VZ * y + z;

        for (int c = 0; c < C; ++c)
            volume[c * vol_cstride + vol_off] = feat_i[c * H * W + feat_off];
    }
}

// Host-side launcher
void LaunchBackprojectCUDA(const float* h_feat, const float* h_proj,
                            float* h_vol, int N, int C, int H, int W,
                            double* t_h2d=nullptr, double* t_kern=nullptr, double* t_d2h=nullptr)
{
    int vol_elems = C * VX * VY * VZ;

    float *d_feat, *d_proj, *d_vol, *d_px, *d_py, *d_pz;
    cudaMalloc(&d_feat, N * C * H * W * sizeof(float));
    cudaMalloc(&d_proj, N * 12 * sizeof(float));
    cudaMalloc(&d_vol, vol_elems * sizeof(float));
    cudaMalloc(&d_px, VX * sizeof(float));
    cudaMalloc(&d_py, VY * sizeof(float));
    cudaMalloc(&d_pz, VZ * sizeof(float));

    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    cudaMemcpy(d_feat, h_feat, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj, h_proj, N * 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_vol, 0, vol_elems * sizeof(float));
    cudaMemcpy(d_px, g_pts.px, VX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, g_pts.py, VY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz, g_pts.pz, VZ * sizeof(float), cudaMemcpyHostToDevice);
    double ms_h2d = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    int total = N * VX * VY;
    int block = 256;
    int grid = (total + block - 1) / block;
    t0 = Clock::now();
    BackprojectKernel<<<grid, block>>>(d_feat, d_proj, d_vol,
                                        d_px, d_py, d_pz, N, C, H, W);
    cudaDeviceSynchronize();
    double ms_kern = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    t0 = Clock::now();
    cudaMemcpy(h_vol, d_vol, vol_elems * sizeof(float), cudaMemcpyDeviceToHost);
    double ms_d2h = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    cudaFree(d_feat); cudaFree(d_proj); cudaFree(d_vol);
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);

    if (t_h2d)  *t_h2d  += ms_h2d;
    if (t_kern) *t_kern += ms_kern;
    if (t_d2h)  *t_d2h  += ms_d2h;
}

}  // anonymous namespace

// ==================================================================
// CUDA pipeline (called from bev_pool.cpp dispatcher when use_gpu=true)
// ==================================================================

cv::Mat FeatureToBEV_CUDA(const cv::Mat& feat_2d,
                           const std::vector<cv::Mat>& extrinsics,
                           int n_sweeps, int n_cams, int feat_w)
{
    using Clock = std::chrono::high_resolution_clock;
    double t_trans = 0, t_proj = 0, t_h2d = 0, t_kern = 0, t_d2h = 0, t_scat = 0;

    int H_feat = feat_2d.size[1];
    int W_feat = feat_2d.size[2];
    int C_feat = feat_2d.size[3];
    int C_total = C_feat * n_sweeps;

    float stride_f = static_cast<float>(feat_w) / W_feat;
    const float* feat_nhwc = reinterpret_cast<const float*>(feat_2d.data);

    int out_elems = VX * VY * VZ * C_total;
    int vol_elems = C_feat * VX * VY * VZ;

    cv::Mat result(1, out_elems, CV_32F);
    float* out = reinterpret_cast<float*>(result.data);

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {

        // 1. NHWC → NCHW (OMP-accelerated shared function)
        auto t0 = Clock::now();
        cv::Mat nchw(1, n_cams * C_feat * H_feat * W_feat, CV_32F);
        float* nchw_ptr = reinterpret_cast<float*>(nchw.data);
        TransposeNHWC2NCHW(feat_nhwc + sweep * n_cams * H_feat * W_feat * C_feat,
                            nchw_ptr, n_cams, H_feat, W_feat, C_feat);
        t_trans += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // 2. Flat projection
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

        // 3. CUDA backprojection
        cv::Mat vol(1, vol_elems, CV_32F);
        t0 = Clock::now();
        LaunchBackprojectCUDA(nchw_ptr, proj,
                               reinterpret_cast<float*>(vol.data),
                               n_cams, C_feat, H_feat, W_feat,
                               &t_h2d, &t_kern, &t_d2h);
        double t_cuda_total = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        const float* vol_ptr = reinterpret_cast<const float*>(vol.data);

        // 4. Scatter into output (OMP-accelerated)
        t0 = Clock::now();
        int cstride_xy = VY * VZ;
        #pragma omp parallel for collapse(2)
        for (int x = 0; x < VX; ++x) {
            for (int y = 0; y < VY; ++y) {
                for (int z = 0; z < VZ; ++z) {
                    int dst_base = x * VY * VZ * C_total + y * VZ * C_total + z * C_total;
                    int src_base = cstride_xy * x + VZ * y + z;
                    for (int c = 0; c < C_feat; ++c) {
                        out[dst_base + sweep * C_feat + c] =
                            vol_ptr[c * VX * VY * VZ + src_base];
                    }
                }
            }
        }
        t_scat += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    int dims[5] = {1, VX, VY, VZ, C_total};
    auto result_5d = cv::Mat(5, dims, CV_32F, out).clone();
    printf("  [CUDA] transpose=%.1f proj=%.1f H2D=%.1f kernel=%.1f D2H=%.1f scatter=%.1f ms\n",
           t_trans, t_proj, t_h2d, t_kern, t_d2h, t_scat);
    return result_5d;
}

}  // namespace bev_pool
