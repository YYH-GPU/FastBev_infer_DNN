#include "nuscenes_loader.h"

#include <fstream>
#include <regex>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace nuscenes {

// ================================================================
// Internal helpers
// ================================================================

static cv::Mat QuatToRotMat(double w, double x, double y, double z)
{
    // pyquaternion convention: q = [w, x, y, z]
    cv::Mat R(3, 3, CV_64F);
    auto* r = (double*)R.data;
    r[0] = w*w + x*x - y*y - z*z;  r[1] = 2*(x*y - w*z);       r[2] = 2*(w*y + x*z);
    r[3] = 2*(x*y + w*z);          r[4] = w*w - x*x + y*y - z*z; r[5] = 2*(y*z - w*x);
    r[6] = 2*(x*z - w*y);          r[7] = 2*(w*x + y*z);          r[8] = w*w - x*x - y*y + z*z;
    return R;
}

static cv::Mat Tfm(const std::vector<double>& translation,
                   const std::vector<double>& rotation_wxyz)
{
    cv::Mat mat = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat R = QuatToRotMat(rotation_wxyz[0], rotation_wxyz[1],
                             rotation_wxyz[2], rotation_wxyz[3]);
    R.copyTo(mat(cv::Rect(0, 0, 3, 3)));
    mat.at<double>(0, 3) = translation[0];
    mat.at<double>(1, 3) = translation[1];
    mat.at<double>(2, 3) = translation[2];
    return mat;
}

static std::string ChannelFromFilename(const std::string& filename)
{
    std::regex re(R"(__((?:CAM(?:_[A-Z]+)+|LIDAR_TOP))__)");
    std::smatch m;
    if (std::regex_search(filename, m, re))
        return m[1].str();
    return "";
}

static std::unordered_map<std::string, json>
KeyframesForSample(const std::string& sample_token, const Tables& tables)
{
    std::unordered_map<std::string, json> kf;
    for (const auto& sd : tables.sample_data_list) {
        if (sd["sample_token"] != sample_token) continue;
        if (!sd["is_key_frame"].get<bool>()) continue;
        std::string ch = ChannelFromFilename(sd["filename"]);
        if (!ch.empty())
            kf[ch] = sd;
    }
    return kf;
}

template<typename T>
static std::vector<T> JsonToVec(const json& j) {
    std::vector<T> v;
    for (const auto& e : j)
        v.push_back(e.get<T>());
    return v;
}

// NuScenes camera_intrinsic is [[fx,0,cx],[0,fy,cy],[0,0,1]] (array of arrays)
static cv::Mat ParseCameraIntrinsic(const json& j) {
    cv::Mat intrin(3, 3, CV_32F);
    int r = 0;
    for (const auto& row : j) {
        int c = 0;
        for (const auto& val : row) {
            intrin.at<float>(r, c) = val.get<float>();
            ++c;
        }
        ++r;
    }
    return intrin;
}

// ================================================================
// Table loading
// ================================================================

Tables LoadTables(const std::string& nuscenes_root)
{
    fs::path meta = fs::path(nuscenes_root) / "v1.0-mini";
    Tables t;

    {
        std::ifstream f(meta / "calibrated_sensor.json");
        json j;
        f >> j;
        for (const auto& r : j)
            t.calibrated_sensor[r["token"].get<std::string>()] = r;
    }
    {
        std::ifstream f(meta / "ego_pose.json");
        json j;
        f >> j;
        for (const auto& r : j)
            t.ego_pose[r["token"].get<std::string>()] = r;
    }
    {
        std::ifstream f(meta / "sample_data.json");
        json j;
        f >> j;
        t.sample_data_list = j.get<std::vector<json>>();
    }
    {
        std::ifstream f(meta / "sample.json");
        json j;
        f >> j;
        t.sample_list = j.get<std::vector<json>>();
        for (const auto& r : t.sample_list)
            t.sample_by_token[r["token"].get<std::string>()] = r;
    }
    return t;
}

// ================================================================
// Temporal tokens
// ================================================================

std::vector<std::string> GetTemporalSampleTokens(
    const std::string& current_token, int n_times, const Tables& tables)
{
    std::vector<std::string> tokens;
    tokens.reserve(n_times);
    std::string tok = current_token;
    for (int i = 0; i < n_times; ++i) {
        tokens.push_back(tok);
        auto it = tables.sample_by_token.find(tok);
        if (it == tables.sample_by_token.end())
            throw std::runtime_error("Sample token not found: " + tok);
        const auto& s = it->second;
        if (i < n_times - 1) {
            if (!s.contains("prev") || s["prev"].is_null() || s["prev"].get<std::string>().empty())
                throw std::runtime_error("Need " + std::to_string(n_times) +
                    " consecutive keyframes; no prev for " + tok);
            tok = s["prev"].get<std::string>();
        }
    }
    return tokens;
}

// ================================================================
// Image paths
// ================================================================

std::unordered_map<std::string, std::string> GetSampleImagePaths(
    const std::string& sample_token, const Tables& tables)
{
    auto kf = KeyframesForSample(sample_token, tables);
    std::unordered_map<std::string, std::string> paths;
    for (const auto& cam : CAMERA_ORDER) {
        if (kf.count(cam))
            paths[cam] = kf[cam]["filename"].get<std::string>();
    }
    return paths;
}

// ================================================================
// Export images
// ================================================================

void ExportTemporalImages(
    const std::vector<std::string>& temporal_tokens,
    const std::string& image_dir,
    const std::string& dataroot,
    const Tables& tables)
{
    fs::path img_dir(image_dir);
    fs::create_directories(img_dir);
    for (size_t sweep_id = 0; sweep_id < temporal_tokens.size(); ++sweep_id) {
        auto paths = GetSampleImagePaths(temporal_tokens[sweep_id], tables);
        for (const auto& cam : CAMERA_ORDER) {
            auto it = paths.find(cam);
            if (it == paths.end()) continue;
            fs::path src = fs::path(dataroot) / it->second;
            fs::path dst = img_dir / (cam + "_sweep" + std::to_string(sweep_id) + ".png");
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        }
    }
}

// ================================================================
// Calibration internals
// ================================================================

static std::pair<cv::Mat, cv::Mat> // (rot, tran)
SensorToLidarRT(const json& cam_cs, const json& cam_ep,
                 const json& lidar_cs, const json& lidar_ep)
{
    auto l2e_r = QuatToRotMat(lidar_cs["rotation"][0], lidar_cs["rotation"][1],
                               lidar_cs["rotation"][2], lidar_cs["rotation"][3]);
    auto e2g_r = QuatToRotMat(lidar_ep["rotation"][0], lidar_ep["rotation"][1],
                               lidar_ep["rotation"][2], lidar_ep["rotation"][3]);
    auto s2e_r = QuatToRotMat(cam_cs["rotation"][0], cam_cs["rotation"][1],
                               cam_cs["rotation"][2], cam_cs["rotation"][3]);
    auto e2g_s_r = QuatToRotMat(cam_ep["rotation"][0], cam_ep["rotation"][1],
                                 cam_ep["rotation"][2], cam_ep["rotation"][3]);

    cv::Mat l2e_t = (cv::Mat_<double>(3, 1)
        << lidar_cs["translation"][0], lidar_cs["translation"][1], lidar_cs["translation"][2]);
    cv::Mat e2g_t = (cv::Mat_<double>(3, 1)
        << lidar_ep["translation"][0], lidar_ep["translation"][1], lidar_ep["translation"][2]);
    cv::Mat s2e_t = (cv::Mat_<double>(3, 1)
        << cam_cs["translation"][0], cam_cs["translation"][1], cam_cs["translation"][2]);
    cv::Mat e2g_s_t = (cv::Mat_<double>(3, 1)
        << cam_ep["translation"][0], cam_ep["translation"][1], cam_ep["translation"][2]);

    // r = (s2e_r.T @ e2g_s_r.T) @ (inv(e2g_r).T @ inv(l2e_r).T)
    cv::Mat s2e_r_t, e2g_s_r_t_m, e2g_r_inv, l2e_r_inv, e2g_r_inv_t, l2e_r_inv_t;
    cv::transpose(s2e_r, s2e_r_t);
    cv::transpose(e2g_s_r, e2g_s_r_t_m);
    e2g_r_inv = e2g_r.inv();
    l2e_r_inv = l2e_r.inv();
    cv::transpose(e2g_r_inv, e2g_r_inv_t);
    cv::transpose(l2e_r_inv, l2e_r_inv_t);

    cv::Mat r_left = s2e_r_t * e2g_s_r_t_m;
    cv::Mat r_right = e2g_r_inv_t * l2e_r_inv_t;
    cv::Mat r = r_left * r_right;

    // t = (s2e_t @ e2g_s_r.T + e2g_s_t) @ (inv(e2g_r).T @ inv(l2e_r).T)
    //   - e2g_t @ (inv(e2g_r).T @ inv(l2e_r).T)
    //   - l2e_t @ inv(l2e_r).T
    cv::Mat s2e_t_t, e2g_s_t_t, e2g_t_t, l2e_t_t;
    cv::transpose(s2e_t, s2e_t_t);
    cv::transpose(e2g_s_t, e2g_s_t_t);
    cv::transpose(e2g_t, e2g_t_t);
    cv::transpose(l2e_t, l2e_t_t);

    cv::Mat term = e2g_r_inv_t * l2e_r_inv_t;
    cv::Mat t = (s2e_t_t * e2g_s_r_t_m + e2g_s_t_t) * term
              - e2g_t_t * term
              - l2e_t_t * l2e_r_inv_t;

    // Return r.T, t (as per Python).  t must be 3x1 column so downstream
    // transpose gives 1x3 for matmul: (1x3) * (3x3) = (1x3).
    cv::Mat r_out, t_out;
    r.convertTo(r_out, CV_32F);
    t.convertTo(t_out, CV_32F);  // t_out is currently 1x3
    cv::Mat r_out_t, t_out_t;
    cv::transpose(r_out, r_out_t);   // r.T (3x3)
    cv::transpose(t_out, t_out_t);   // t: 1x3 → 3x1 column vector
    return {r_out_t, t_out_t};
}

static cv::Mat LidarAdj2LidarCurr(
    const std::string& adj_token, const std::string& curr_token, const Tables& tables)
{
    const auto& cals = tables.calibrated_sensor;
    const auto& ego_poses = tables.ego_pose;

    auto lidar_ego = [&](const std::string& st) -> std::pair<cv::Mat, cv::Mat> {
        auto kf = KeyframesForSample(st, tables);
        const auto& sd = kf.at("LIDAR_TOP");
        const auto& cs = cals.at(sd["calibrated_sensor_token"].get<std::string>());
        const auto& ep = ego_poses.at(sd["ego_pose_token"].get<std::string>());
        auto cs_t = JsonToVec<double>(cs["translation"]);
        auto cs_r = JsonToVec<double>(cs["rotation"]);
        auto ep_t = JsonToVec<double>(ep["translation"]);
        auto ep_r = JsonToVec<double>(ep["rotation"]);
        return {Tfm(cs_t, cs_r), Tfm(ep_t, ep_r)};
    };

    auto [l2e_curr, e2g_curr] = lidar_ego(curr_token);
    auto [l2e_adj, e2g_adj] = lidar_ego(adj_token);

    cv::Mat l2e_curr_inv = l2e_curr.inv();
    cv::Mat e2g_curr_inv = e2g_curr.inv();
    cv::Mat warp = l2e_curr_inv * e2g_curr_inv * e2g_adj * l2e_adj;
    cv::Mat warp32;
    warp.convertTo(warp32, CV_32F);
    return warp32;
}

static std::pair<cv::Mat, cv::Mat> ApplyLidarWarp(
    const cv::Mat& rot, const cv::Mat& tran, const cv::Mat& warp_4x4)
{
    cv::Mat mat = cv::Mat::eye(4, 4, CV_32F);
    rot.copyTo(mat(cv::Rect(0, 0, 3, 3)));
    tran.copyTo(mat(cv::Rect(3, 0, 1, 3)));
    mat = warp_4x4 * mat;
    return {mat(cv::Rect(0, 0, 3, 3)).clone(), mat(cv::Rect(3, 0, 1, 3)).clone()};
}

// ================================================================
// Public calibration API
// ================================================================

std::unordered_map<std::string, cv::Mat>
LoadSampleCalibNative(const std::string& sample_token, const Tables& tables)
{
    const auto& cals = tables.calibrated_sensor;
    const auto& ego_poses = tables.ego_pose;
    auto kf = KeyframesForSample(sample_token, tables);
    const auto& lidar_sd = kf.at("LIDAR_TOP");
    const auto& lidar_cs = cals.at(lidar_sd["calibrated_sensor_token"].get<std::string>());
    const auto& lidar_ep = ego_poses.at(lidar_sd["ego_pose_token"].get<std::string>());

    std::unordered_map<std::string, cv::Mat> lidar2img_native;

    for (const auto& cam : CAMERA_ORDER) {
        const auto& cam_sd = kf.at(cam);
        const auto& cam_cs_j = cals.at(cam_sd["calibrated_sensor_token"].get<std::string>());
        const auto& cam_ep_j = ego_poses.at(cam_sd["ego_pose_token"].get<std::string>());

        auto [rot, tran] = SensorToLidarRT(cam_cs_j, cam_ep_j, lidar_cs, lidar_ep);

        // Build 3x4 projection matrix
        cv::Mat lidar2cam_r = rot.inv();
        cv::Mat tran_t, lidar2cam_r_t2;
        cv::transpose(tran, tran_t);
        cv::transpose(lidar2cam_r, lidar2cam_r_t2);
        cv::Mat lidar2cam_t = tran_t * lidar2cam_r_t2;  // 1x3

        cv::Mat lidar2cam_rt = cv::Mat::eye(4, 4, CV_32F);
        lidar2cam_r_t2.copyTo(lidar2cam_rt(cv::Rect(0, 0, 3, 3)));
        lidar2cam_rt.at<float>(3, 0) = -lidar2cam_t.at<float>(0, 0);
        lidar2cam_rt.at<float>(3, 1) = -lidar2cam_t.at<float>(0, 1);
        lidar2cam_rt.at<float>(3, 2) = -lidar2cam_t.at<float>(0, 2);

        cv::Mat viewpad = cv::Mat::eye(4, 4, CV_32F);
        cv::Mat intrin = ParseCameraIntrinsic(cam_cs_j["camera_intrinsic"]);
        intrin.copyTo(viewpad(cv::Rect(0, 0, 3, 3)));

        cv::Mat lidar2cam_rt_t;
        cv::transpose(lidar2cam_rt, lidar2cam_rt_t);
        cv::Mat proj = viewpad * lidar2cam_rt_t;
        lidar2img_native[cam] = proj(cv::Rect(0, 0, 4, 3)).clone();
    }
    return lidar2img_native;
}

std::unordered_map<std::string, CamCalibAug>
LoadSampleCalibAug(const std::string& sample_token,
                    const std::string& ref_sample_token,
                    const Tables& tables)
{
    const auto& cals = tables.calibrated_sensor;
    const auto& ego_poses = tables.ego_pose;

    cv::Mat warp;
    if (ref_sample_token != sample_token && !ref_sample_token.empty()) {
        warp = LidarAdj2LidarCurr(sample_token, ref_sample_token, tables);
    }

    auto kf = KeyframesForSample(sample_token, tables);
    const auto& lidar_sd = kf.at("LIDAR_TOP");
    const auto& lidar_cs = cals.at(lidar_sd["calibrated_sensor_token"].get<std::string>());
    const auto& lidar_ep = ego_poses.at(lidar_sd["ego_pose_token"].get<std::string>());

    std::unordered_map<std::string, CamCalibAug> result;

    for (const auto& cam : CAMERA_ORDER) {
        const auto& cam_sd = kf.at(cam);
        const auto& cam_cs_j = cals.at(cam_sd["calibrated_sensor_token"].get<std::string>());
        const auto& cam_ep_j = ego_poses.at(cam_sd["ego_pose_token"].get<std::string>());

        auto [rot, tran] = SensorToLidarRT(cam_cs_j, cam_ep_j, lidar_cs, lidar_ep);

        if (!warp.empty())
            std::tie(rot, tran) = ApplyLidarWarp(rot, tran, warp);

        CamCalibAug calib;
        calib.intrin = ParseCameraIntrinsic(cam_cs_j["camera_intrinsic"]);
        calib.rot = rot.clone();
        calib.tran = tran.clone();
        result[cam] = calib;
    }
    return result;
}

// ================================================================
// Augmentation
// ================================================================

AugParams SampleTestAugmentation(int height, int width, const ImageAugCfg& cfg)
{
    AugParams aug;
    int fW = cfg.test_input_size.width;
    int fH = cfg.test_input_size.height;

    aug.resize = static_cast<float>(fW) / static_cast<float>(width) + cfg.test_resize;
    int newW = static_cast<int>(width * aug.resize);
    int newH = static_cast<int>(height * aug.resize);
    aug.resize_dims = cv::Size(newW, newH);

    int crop_w_start = (newW - fW) / 2;
    int crop_h_start = (newH - fH) / 2;
    aug.crop = cv::Rect(crop_w_start, crop_h_start, fW, fH);

    aug.flip = cfg.test_flip;
    aug.rotate = cfg.test_rotate;
    aug.fH = fH;
    aug.fW = fW;
    return aug;
}

std::pair<cv::Mat, cv::Mat> ImageTransformPostMatrix(const AugParams& aug)
{
    cv::Mat post_rot = cv::Mat::eye(2, 2, CV_32F);
    cv::Mat post_tran = cv::Mat::zeros(2, 1, CV_32F);

    post_rot *= aug.resize;
    post_tran.at<float>(0, 0) = -static_cast<float>(aug.crop.x);
    post_tran.at<float>(1, 0) = -static_cast<float>(aug.crop.y);

    if (aug.flip) {
        float a_data[] = {-1.f, 0.f, 0.f, 1.f};
        cv::Mat a(2, 2, CV_32F, a_data);
        float b_data[] = {static_cast<float>(aug.crop.x + aug.crop.width), 0.f};
        cv::Mat b(2, 1, CV_32F, b_data);
        post_rot = a * post_rot;
        post_tran = a * post_tran + b;
    }

    if (std::abs(aug.rotate) > 1e-6f) {
        float h = aug.rotate / 180.f * CV_PI;
        float ca = std::cos(h), sa = std::sin(h);
        float a_data[] = {ca, sa, -sa, ca};
        cv::Mat a(2, 2, CV_32F, a_data);
        float bx = (aug.crop.x + aug.crop.width) * 0.5f;
        float by = (aug.crop.y + aug.crop.height) * 0.5f;
        float b_data[] = {bx, by};
        cv::Mat b(2, 1, CV_32F, b_data);
        b = a * (-b) + b;
        post_rot = a * post_rot;
        post_tran = a * post_tran + b;
    }

    cv::Mat ret_post_rot = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat ret_post_tran = cv::Mat::zeros(3, 1, CV_32F);
    post_rot.copyTo(ret_post_rot(cv::Rect(0, 0, 2, 2)));
    post_tran.copyTo(ret_post_tran(cv::Rect(0, 0, 1, 2)));
    return {ret_post_rot, ret_post_tran};
}

// ================================================================
// Projection
// ================================================================

cv::Mat Rts2Proj(const CamCalibAug& cam_info,
                 const cv::Mat& post_rot, const cv::Mat& post_tran)
{
    cv::Mat lidar2cam_r = cam_info.rot.inv();
    cv::Mat tran_t, lidar2cam_r_t;
    cv::transpose(cam_info.tran, tran_t);
    cv::transpose(lidar2cam_r, lidar2cam_r_t);
    cv::Mat lidar2cam_t = tran_t * lidar2cam_r_t;

    cv::Mat lidar2cam_rt = cv::Mat::eye(4, 4, CV_32F);
    lidar2cam_r_t.copyTo(lidar2cam_rt(cv::Rect(0, 0, 3, 3)));
    lidar2cam_rt.at<float>(3, 0) = -lidar2cam_t.at<float>(0, 0);
    lidar2cam_rt.at<float>(3, 1) = -lidar2cam_t.at<float>(0, 1);
    lidar2cam_rt.at<float>(3, 2) = -lidar2cam_t.at<float>(0, 2);

    cv::Mat viewpad = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat top_part = post_rot * cam_info.intrin;
    top_part.copyTo(viewpad(cv::Rect(0, 0, 3, 3)));
    viewpad.at<float>(0, 2) += post_tran.at<float>(0, 0);
    viewpad.at<float>(1, 2) += post_tran.at<float>(1, 0);
    viewpad.at<float>(2, 2) += post_tran.at<float>(2, 0);

    cv::Mat lidar2cam_rt_t;
    cv::transpose(lidar2cam_rt, lidar2cam_rt_t);
    cv::Mat result = viewpad * lidar2cam_rt_t;
    return result.clone();
}

std::vector<cv::Mat> BuildTemporalExtrinsics(
    const std::vector<std::string>& temporal_tokens,
    int height, int width,
    const Tables& tables,
    AugParams* out_aug,
    const ImageAugCfg& data_cfg)
{
    const std::string& curr = temporal_tokens[0];
    AugParams aug = SampleTestAugmentation(height, width, data_cfg);
    auto [post_rot, post_tran] = ImageTransformPostMatrix(aug);

    if (out_aug) *out_aug = aug;

    std::vector<cv::Mat> extrinsics;
    for (const auto& tok : temporal_tokens) {
        std::string ref = (tok != curr) ? curr : "";
        auto calibs = LoadSampleCalibAug(tok, ref, tables);
        for (const auto& cam : CAMERA_ORDER) {
            extrinsics.push_back(Rts2Proj(calibs.at(cam), post_rot, post_tran));
        }
    }
    return extrinsics;
}

}  // namespace nuscenes
