#include "visualize.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

namespace visualize {

const std::vector<std::string> CATEGORIES = {
    "car","truck","trailer","bus","construction_vehicle",
    "bicycle","motorcycle","pedestrian","traffic_cone","barrier"
};

const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(31, 119, 180), cv::Scalar(255, 127, 14),
    cv::Scalar(44, 160, 44),  cv::Scalar(148, 103, 189),
    cv::Scalar(140, 86, 75),  cv::Scalar(227, 119, 194),
    cv::Scalar(127, 127, 127),cv::Scalar(188, 189, 34),
    cv::Scalar(23, 190, 207), cv::Scalar(174, 199, 232)
};

static std::vector<cv::Point2f> BEVCorners(float cx, float cy, float w, float l, float yaw)
{
    float ca = std::cos(yaw), sa = std::sin(yaw);
    float hw = w * 0.5f, hl = l * 0.5f;
    return {
        {hw * ca - hl * sa + cx, hw * sa + hl * ca + cy},
        {hw * ca + hl * sa + cx, hw * sa - hl * ca + cy},
        {-hw * ca + hl * sa + cx, -hw * sa - hl * ca + cy},
        {-hw * ca - hl * sa + cx, -hw * sa + hl * ca + cy},
    };
}

cv::Mat DrawBEVFrame(const std::vector<Detection>& dets, float score_thr)
{
    const int size = 1000;
    const float range = 50.0f;
    cv::Mat bev(size, size, CV_8UC3, cv::Scalar(245, 245, 245));

    // Grid
    for (int i = -50; i <= 50; i += 10) {
        int px = static_cast<int>((i + range) / (2 * range) * size);
        cv::line(bev, cv::Point(px, 0), cv::Point(px, size), cv::Scalar(220, 220, 220), 1);
        cv::line(bev, cv::Point(0, px), cv::Point(size, px), cv::Scalar(220, 220, 220), 1);
    }

    // Origin crosshair
    int cx = size / 2, cy = size / 2;
    cv::line(bev, cv::Point(cx, 0), cv::Point(cx, size), cv::Scalar(180, 180, 180), 1);
    cv::line(bev, cv::Point(0, cy), cv::Point(size, cy), cv::Scalar(180, 180, 180), 1);

    // Ego vehicle
    std::vector<cv::Point> ego = {
        {cx - 10, cy - 25}, {cx + 10, cy - 25},
        {cx + 10, cy + 25}, {cx - 10, cy + 25}
    };
    cv::fillPoly(bev, std::vector<std::vector<cv::Point>>{ego}, cv::Scalar(50, 50, 50));

    for (const auto& d : dets) {
        if (d.score < score_thr) continue;
        if (std::abs(d.x) > 55 || std::abs(d.y) > 55) continue;

        auto corners_lidar = BEVCorners(d.x, d.y, d.w, d.l, d.yaw);
        std::vector<cv::Point> pts;
        float s = size / (2.0f * range);
        for (const auto& c : corners_lidar) {
            int px = static_cast<int>(c.x * s + size / 2);
            int py = static_cast<int>(size - (c.y * s + size / 2));
            pts.push_back(cv::Point(px, py));
        }

        cv::Scalar clr = COLORS[d.label % 10];
        float alpha = std::min(1.0f, d.score * 2.0f);
        cv::Scalar blended(
            (1-alpha)*245 + alpha*clr[0],
            (1-alpha)*245 + alpha*clr[1],
            (1-alpha)*245 + alpha*clr[2]
        );

        cv::fillConvexPoly(bev, pts, blended);
        cv::polylines(bev, pts, true, clr * 0.7, 1);
    }

    // Legend
    int ly = 15;
    for (int i = 0; i < 10; ++i) {
        cv::rectangle(bev, cv::Rect(5, ly, 10, 10), COLORS[i], -1);
        cv::putText(bev, CATEGORIES[i], cv::Point(18, ly + 9),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 1);
        ly += 14;
    }

    // Title
    cv::putText(bev, "FastBEV ONNX - BEV Detections", cv::Point(250, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    return bev;
}

static void DrawBEV(const std::vector<Detection>& dets, const std::string& out_path,
                     float score_thr)
{
    cv::Mat bev = DrawBEVFrame(dets, score_thr);
    cv::imwrite(out_path, bev);
    printf("Saved BEV detections -> %s\n", out_path.c_str());
}

// 8 corners of a 3D box
static cv::Mat GetBoxCorners3D(const Detection& d)
{
    cv::Mat corners(8, 3, CV_32F);
    float ca = std::cos(d.yaw), sa = std::sin(d.yaw);
    float hw = d.w * 0.5f, hl = d.l * 0.5f;
    float dx[8] = {-hw, -hw, -hw, -hw, hw, hw, hw, hw};
    float dy[8] = {-hl, -hl, hl, hl, -hl, -hl, hl, hl};
    float dz[8] = {0.f, d.h, d.h, 0.f, 0.f, d.h, d.h, 0.f};
    for (int i = 0; i < 8; ++i) {
        corners.at<float>(i, 0) = ca * dx[i] - sa * dy[i] + d.x;
        corners.at<float>(i, 1) = sa * dx[i] + ca * dy[i] + d.y;
        corners.at<float>(i, 2) = dz[i] + d.z;
    }
    return corners;
}

static const int BOX_EDGES[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
};

static void ProjectAndDrawBoxes(
    const std::vector<Detection>& dets,
    const std::vector<cv::Mat>& images,
    const std::vector<std::string>& cam_names,
    const std::unordered_map<std::string, cv::Mat>& lidar2img_native,
    const std::string& out_path,
    float score_thr)
{
    int n_cams = (int)images.size();
    if (n_cams == 0) return;

    // Build grid: 2 rows, 3 cols
    int grid_cols = 3, grid_rows = 2;
    int cell_w = 800, cell_h = 450;
    cv::Mat canvas(grid_rows * cell_h, grid_cols * cell_w, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int ci = 0; ci < n_cams; ++ci) {
        int gx = ci % grid_cols, gy = ci / grid_cols;
        cv::Rect roi(gx * cell_w, gy * cell_h, cell_w, cell_h);

        cv::Mat img;
        cv::resize(images[ci], img, cv::Size(cell_w, cell_h));
        if (img.channels() == 3)
            img.copyTo(canvas(roi));
        else if (img.channels() == 1)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        img.copyTo(canvas(roi));

        // Draw camera name
        cv::putText(canvas, cam_names[ci], cv::Point(gx * cell_w + 10, gy * cell_h + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        // Project 3D boxes
        auto it = lidar2img_native.find(cam_names[ci]);
        if (it == lidar2img_native.end()) continue;
        cv::Mat lidar2img = it->second;  // 3x4

        float scale_x = (float)cell_w / 1600.0f;
        float scale_y = (float)cell_h / 900.0f;

        for (const auto& d : dets) {
            if (d.score < score_thr) continue;
            cv::Scalar clr = COLORS[d.label % 10];
            float alpha = std::min(1.0f, d.score * 2.0f);

            auto corners = GetBoxCorners3D(d);
            // Project each corner
            std::vector<cv::Point2f> uv(8);
            std::vector<bool> vis(8, false);
            for (int k = 0; k < 8; ++k) {
                float cx = corners.at<float>(k, 0);
                float cy = corners.at<float>(k, 1);
                float cz = corners.at<float>(k, 2);
                float w = lidar2img.at<float>(2, 0)*cx +
                         lidar2img.at<float>(2, 1)*cy +
                         lidar2img.at<float>(2, 2)*cz +
                         lidar2img.at<float>(2, 3);
                if (w <= 0) continue;
                float uw = lidar2img.at<float>(0, 0)*cx +
                          lidar2img.at<float>(0, 1)*cy +
                          lidar2img.at<float>(0, 2)*cz +
                          lidar2img.at<float>(0, 3);
                float vw = lidar2img.at<float>(1, 0)*cx +
                          lidar2img.at<float>(1, 1)*cy +
                          lidar2img.at<float>(1, 2)*cz +
                          lidar2img.at<float>(1, 3);
                uv[k].x = uw / w * scale_x + gx * cell_w;
                uv[k].y = vw / w * scale_y + gy * cell_h;
                vis[k] = (uv[k].x >= gx*cell_w && uv[k].x < (gx+1)*cell_w &&
                          uv[k].y >= gy*cell_h && uv[k].y < (gy+1)*cell_h);
            }

            for (const auto& e : BOX_EDGES) {
                if (vis[e[0]] && vis[e[1]]) {
                    cv::Scalar blended(
                        (1-alpha)*255 + alpha*clr[0],
                        (1-alpha)*255 + alpha*clr[1],
                        (1-alpha)*255 + alpha*clr[2]
                    );
                    cv::line(canvas, uv[e[0]], uv[e[1]], blended, 2);
                }
            }
        }
    }

    cv::imwrite(out_path, canvas);
    printf("Saved camera 3D boxes -> %s\n", out_path.c_str());
}

static cv::Mat DrawCameraGrid(const std::vector<cv::Mat>& images,
                               const std::vector<std::string>& cam_names)
{
    int n = (int)images.size();
    int cols = 3, rows = 2;
    int cw = 800, ch = 450;
    cv::Mat canvas(rows * ch, cols * cw, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < n; ++i) {
        int gx = i % cols, gy = i / cols;
        cv::Mat resized;
        cv::resize(images[i], resized, cv::Size(cw, ch));
        resized.copyTo(canvas(cv::Rect(gx * cw, gy * ch, cw, ch)));
        cv::putText(canvas, cam_names[i], cv::Point(gx * cw + 10, gy * ch + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
    return canvas;
}

cv::Mat DrawCombinedFrame(
    const std::vector<Detection>& dets,
    const std::vector<cv::Mat>& cam_images,
    const std::vector<std::string>& cam_names,
    const std::unordered_map<std::string, cv::Mat>& lidar2img_native,
    float score_thr)
{
    // Layout (3-row):
    //   Top:    CAM_FRONT_LEFT  |  CAM_FRONT  |  CAM_FRONT_RIGHT
    //   Middle:                  BEV view
    //   Bottom: CAM_BACK_LEFT   |  CAM_BACK   |  CAM_BACK_RIGHT

    const int canvas_w = 1800, canvas_h = 1200;
    const int cell_w = 540, cell_h = 304;  // 16:9
    const int gap = 20;
    const int top_y = 0;
    const int bottom_y_actual = canvas_h - cell_h;

    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(30, 30, 30));

    // Display order: top 3 then bottom 3
    const std::vector<std::string> display_order = {
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",  "CAM_BACK", "CAM_BACK_RIGHT"
    };

    // Build name→image map
    std::unordered_map<std::string, cv::Mat> name_to_img;
    for (size_t i = 0; i < cam_names.size(); ++i)
        name_to_img[cam_names[i]] = cam_images[i];

    // Helper: draw one camera cell with 3D boxes
    auto drawCameraCell = [&](const std::string& cam_name, int col, int row_y) {
        auto it = name_to_img.find(cam_name);
        if (it == name_to_img.end() || it->second.empty()) return;

        cv::Mat img = it->second;
        int orig_w = img.cols, orig_h = img.rows;
        if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        double scale = std::min((double)cell_w / orig_w, (double)cell_h / orig_h);
        int new_w = (int)(orig_w * scale);
        int new_h = (int)(orig_h * scale);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h));

        int off_x = col * cell_w + (cell_w - new_w) / 2;
        int off_y = row_y + (cell_h - new_h) / 2;
        cv::Rect dst_roi(off_x, off_y, new_w, new_h);
        if (dst_roi.x + dst_roi.width <= canvas_w && dst_roi.y + dst_roi.height <= canvas_h)
            resized.copyTo(canvas(dst_roi));

        // Camera name
        std::string label = cam_name;
        if (label.rfind("CAM_", 0) == 0) label = label.substr(4);  // strip "CAM_"
        cv::putText(canvas, label, cv::Point(col * cell_w + 5, row_y + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);

        // --- Project 3D boxes ---
        auto calib_it = lidar2img_native.find(cam_name);
        if (calib_it == lidar2img_native.end()) return;
        cv::Mat lidar2img = calib_it->second;

        float proj_scale_x = (float)scale;
        float proj_scale_y = (float)scale;

        for (const auto& d : dets) {
            if (d.score < score_thr) continue;
            cv::Scalar clr = COLORS[d.label % 10];
            float alpha = std::min(1.0f, d.score * 2.0f);

            float ca = std::cos(d.yaw), sa = std::sin(d.yaw);
            float hw = d.w * 0.5f, hl = d.l * 0.5f;
            float dx[8] = {-hw, -hw, -hw, -hw, hw, hw, hw, hw};
            float dy[8] = {-hl, -hl, hl, hl, -hl, -hl, hl, hl};
            float dz[8] = {0.f, d.h, d.h, 0.f, 0.f, d.h, d.h, 0.f};

            std::vector<cv::Point2f> uv(8);
            std::vector<bool> vis(8, false);
            for (int k = 0; k < 8; ++k) {
                float cx = ca * dx[k] - sa * dy[k] + d.x;
                float cy = sa * dx[k] + ca * dy[k] + d.y;
                float cz = dz[k] + d.z;
                float w = lidar2img.at<float>(2, 0) * cx +
                         lidar2img.at<float>(2, 1) * cy +
                         lidar2img.at<float>(2, 2) * cz +
                         lidar2img.at<float>(2, 3);
                if (w <= 0) continue;
                float uw = lidar2img.at<float>(0, 0) * cx +
                          lidar2img.at<float>(0, 1) * cy +
                          lidar2img.at<float>(0, 2) * cz +
                          lidar2img.at<float>(0, 3);
                float vw = lidar2img.at<float>(1, 0) * cx +
                          lidar2img.at<float>(1, 1) * cy +
                          lidar2img.at<float>(1, 2) * cz +
                          lidar2img.at<float>(1, 3);
                uv[k].x = uw / w * proj_scale_x + off_x;
                uv[k].y = vw / w * proj_scale_y + off_y;
                vis[k] = (uv[k].x >= col * cell_w && uv[k].x < (col + 1) * cell_w &&
                          uv[k].y >= row_y && uv[k].y < row_y + cell_h);
            }

            const int edges[12][2] = {
                {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
            };
            for (const auto& e : edges) {
                if (vis[e[0]] && vis[e[1]]) {
                    cv::Scalar blended(
                        (1 - alpha) * 255 + alpha * clr[0],
                        (1 - alpha) * 255 + alpha * clr[1],
                        (1 - alpha) * 255 + alpha * clr[2]);
                    cv::line(canvas, uv[e[0]], uv[e[1]], blended, 2);
                }
            }
        }
    };

    // Draw top row: FRONT_LEFT, FRONT, FRONT_RIGHT
    for (int col = 0; col < 3; ++col)
        drawCameraCell(display_order[col], col, top_y);

    // Draw bottom row: BACK_LEFT, BACK, BACK_RIGHT
    for (int col = 0; col < 3; ++col)
        drawCameraCell(display_order[3 + col], col, bottom_y_actual);

    // ---- BEV in the middle ----
    int bev_size = canvas_h - cell_h * 2 - gap * 2;               // available height
    // make it square, at most 800px
    if (bev_size > 800) bev_size = 800;
    int bev_x = (canvas_w - bev_size) / 2;
    int bev_y_actual = (canvas_h - bev_size) / 2;

    cv::Mat bev = DrawBEVFrame(dets, score_thr);
    cv::Mat bev_resized;
    cv::resize(bev, bev_resized, cv::Size(bev_size, bev_size));
    bev_resized.copyTo(canvas(cv::Rect(bev_x, bev_y_actual, bev_size, bev_size)));

    // Separator lines between camera rows and BEV
    int sep_top = top_y + cell_h + gap / 2;
    int sep_bot = bottom_y_actual - gap / 2;
    cv::line(canvas, cv::Point(0, sep_top), cv::Point(canvas_w, sep_top),
             cv::Scalar(80, 80, 80), 1);
    cv::line(canvas, cv::Point(0, sep_bot), cv::Point(canvas_w, sep_bot),
             cv::Scalar(80, 80, 80), 1);

    return canvas;
}

void VisualizeResults(
    const std::vector<Detection>& detections,
    const std::string& image_dir,
    const std::string& out_dir,
    const std::unordered_map<std::string, cv::Mat>& lidar2img_native,
    const std::vector<std::string>& camera_order,
    float score_thr)
{
    fs::create_directories(out_dir);

    // Filter by score
    std::vector<Detection> filtered;
    for (const auto& d : detections) {
        if (d.score >= score_thr)
            filtered.push_back(d);
    }

    // 1. BEV detections
    DrawBEV(filtered, out_dir + "/bev_detections.png", score_thr);

    // 2. Load original camera images (current sweep only)
    std::vector<cv::Mat> orig_imgs;
    std::vector<std::string> cam_names;
    for (const auto& cam : camera_order) {
        for (const auto& ext : {".png", ".jpg", ".jpeg"}) {
            std::string path = image_dir + "/" + cam + "_sweep0" + ext;
            cv::Mat img = cv::imread(path);
            if (!img.empty()) {
                orig_imgs.push_back(img);
                cam_names.push_back(cam);
                break;
            }
        }
    }

    // Save camera grid
    cv::Mat cam_grid = DrawCameraGrid(orig_imgs, cam_names);
    cv::imwrite(out_dir + "/camera_inputs.png", cam_grid);
    printf("Saved camera inputs -> %s/camera_inputs.png\n", out_dir.c_str());

    // 3. Camera 3D boxes
    ProjectAndDrawBoxes(detections, orig_imgs, cam_names,
                        lidar2img_native,
                        out_dir + "/camera_3d_boxes.png",
                        score_thr);
}

}  // namespace visualize
