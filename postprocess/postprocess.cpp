#include "postprocess.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace postprocess {

// ---- Constants ----
const float ANCHOR_SIZES[4][3] = {
    {0.8660f, 2.5981f, 1.0f},
    {0.5774f, 1.7321f, 1.0f},
    {1.0f,    1.0f,    1.0f},
    {0.4f,    0.4f,    1.0f},
};
const float ANCHOR_ROTATIONS[2] = {0.0f, 1.57f};
const float ANCHOR_RANGE[6] = {-50, -50, -1.8f, 50, 50, -1.8f};
const int NMS_PRE = 200;
const int MAX_NUM = 50;
const float NMS_SCORE_THR = 0.3f;
const std::vector<std::string> NMS_TYPE_LIST = {
    "rotate","rotate","rotate","rotate","rotate","rotate","rotate","rotate","rotate","circle"
};
const float NMS_THR_LIST[10] = {0.2f,0.2f,0.2f,0.2f,0.2f,0.2f,0.2f,0.5f,0.5f,0.2f};
const float NMS_RADIUS_THR_LIST[10] = {4.0f,12.0f,10.0f,10.0f,12.0f,0.85f,0.85f,0.175f,0.175f,1.0f};
const float NMS_RESCALE_FACTOR[10] = {1.0f,0.7f,0.55f,0.4f,0.7f,1.0f,1.0f,4.5f,9.0f,1.0f};
const float DIR_OFFSET = 0.7854f;  // pi/4

// ================================================================
// Anchor generation
// ================================================================

cv::Mat GenerateAnchors(int feat_h, int feat_w)
{
    int H = feat_h, W = feat_w;
    int fsize[3] = {1, H, W};
    int num_sizes = 4;
    int num_rots = 2;

    const auto& ar = ANCHOR_RANGE;

    std::vector<float> z_centers(fsize[0]);
    std::vector<float> y_centers(fsize[1]);
    std::vector<float> x_centers(fsize[2]);

    for (int i = 0; i < fsize[0]; ++i)
        z_centers[i] = ar[2] + (ar[5] - ar[2]) * (i + 0.5f) / fsize[0];
    for (int i = 0; i < fsize[1]; ++i)
        y_centers[i] = ar[1] + (ar[4] - ar[1]) * (i + 0.5f) / fsize[1];
    for (int i = 0; i < fsize[2]; ++i)
        x_centers[i] = ar[0] + (ar[3] - ar[0]) * (i + 0.5f) / fsize[2];

    // Order: z -> y -> x -> size -> rot  (aligned with reshape)
    // Final reshape: [1, H, W, N, R, 7] → [-1, 7] → pad to [-1, 9]
    int total = fsize[0] * fsize[1] * fsize[2] * num_sizes * num_rots; // 1*100*100*4*2 = 80000
    cv::Mat anchors(total, 9, CV_32F);

    int idx = 0;
    for (int z = 0; z < fsize[0]; ++z) {
        float zc = z_centers[z];
        for (int y = 0; y < fsize[1]; ++y) {
            float yc = y_centers[y];
            for (int x = 0; x < fsize[2]; ++x) {
                float xc = x_centers[x];
                for (int s = 0; s < num_sizes; ++s) {
                    float w = ANCHOR_SIZES[s][0];
                    float l = ANCHOR_SIZES[s][1];
                    float h = ANCHOR_SIZES[s][2];
                    for (int r = 0; r < num_rots; ++r) {
                        float rot = ANCHOR_ROTATIONS[r];
                        auto* row = anchors.ptr<float>(idx);
                        row[0] = xc;
                        row[1] = yc;
                        row[2] = zc;
                        row[3] = w;
                        row[4] = l;
                        row[5] = h;
                        row[6] = rot;
                        row[7] = 0; // vx
                        row[8] = 0; // vy
                        ++idx;
                    }
                }
            }
        }
    }
    return anchors;
}

// ================================================================
// Box decoding (DeltaXYZWLHRBBoxCoder)
// ================================================================

cv::Mat DecodeBoxes(const cv::Mat& anchors, const cv::Mat& deltas)
{
    int N = anchors.rows;
    CV_Assert(deltas.rows == N);
    cv::Mat boxes(N, 9, CV_32F);

    for (int i = 0; i < N; ++i) {
        const float* a = anchors.ptr<float>(i);
        const float* d = deltas.ptr<float>(i);
        float* b = boxes.ptr<float>(i);

        float xa = a[0], ya = a[1], za = a[2], wa = a[3], la = a[4], ha = a[5], ra = a[6];
        float xt = d[0], yt = d[1], zt = d[2], wt = d[3], lt = d[4], ht = d[5], rt = d[6];

        za = za + ha / 2;
        float diagonal = std::sqrt(la * la + wa * wa);

        float xg = xt * diagonal + xa;
        float yg = yt * diagonal + ya;
        float zg = zt * ha + za;
        float lg = std::exp(lt) * la;
        float wg = std::exp(wt) * wa;
        float hg = std::exp(ht) * ha;
        float rg = rt + ra;
        zg = zg - hg / 2;

        b[0] = xg; b[1] = yg; b[2] = zg;
        b[3] = wg; b[4] = lg; b[5] = hg; b[6] = rg;
        b[7] = d[7] + a[7];
        b[8] = d[8] + a[8];
    }
    return boxes;
}

// ================================================================
// Circle NMS
// ================================================================

std::vector<int> CircleNMS(const cv::Mat& dets, float thresh)
{
    int ndets = dets.rows;
    std::vector<int> order(ndets);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return dets.at<float>(a, 2) > dets.at<float>(b, 2);
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(ndets, false);

    for (int i = 0; i < ndets; ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        float ix = dets.at<float>(idx, 0);
        float iy = dets.at<float>(idx, 1);
        for (int j = i + 1; j < ndets; ++j) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            float dx = ix - dets.at<float>(jdx, 0);
            float dy = iy - dets.at<float>(jdx, 1);
            if (dx * dx + dy * dy <= thresh)
                suppressed[jdx] = true;
        }
    }
    return keep;
}

// ================================================================
// Rotated IoU (Sutherland-Hodgman based)
// ================================================================

static cv::Mat GetBEVCorners(float cx, float cy, float w, float l, float r)
{
    float ca = std::cos(r), sa = std::sin(r);
    float hw = w * 0.5f, hl = l * 0.5f;
    cv::Mat corners(4, 2, CV_32F);
    float lc[4][2] = {{-hw, -hl}, {hw, -hl}, {hw, hl}, {-hw, hl}};
    for (int i = 0; i < 4; ++i) {
        corners.at<float>(i, 0) = lc[i][0] * ca - lc[i][1] * sa + cx;
        corners.at<float>(i, 1) = lc[i][0] * sa + lc[i][1] * ca + cy;
    }
    return corners;
}

static cv::Point2f LineIntersection(cv::Point2f p1, cv::Point2f p2,
                                     cv::Point2f p3, cv::Point2f p4)
{
    cv::Point2f d1 = p2 - p1;
    cv::Point2f d2 = p4 - p3;
    float det = d1.x * d2.y - d1.y * d2.x;
    if (std::abs(det) < 1e-10f) return p1;
    float t = ((p3.x - p1.x) * d2.y - (p3.y - p1.y) * d2.x) / det;
    return p1 + t * d1;
}

static float Cross2D(cv::Point2f a, cv::Point2f b)
{
    return a.x * b.y - a.y * b.x;
}

static std::vector<cv::Point2f> PolygonIntersection(
    const std::vector<cv::Point2f>& subject,
    const std::vector<cv::Point2f>& clip)
{
    std::vector<cv::Point2f> output = subject;
    int n_clip = (int)clip.size();

    for (int i = 0; i < n_clip; ++i) {
        if (output.empty()) return {};
        cv::Point2f e1 = clip[i], e2 = clip[(i + 1) % n_clip];
        cv::Point2f edge_vec = e2 - e1;
        std::vector<cv::Point2f> new_out;
        int n_out = (int)output.size();

        for (int j = 0; j < n_out; ++j) {
            cv::Point2f cur = output[j];
            cv::Point2f prev = output[(j - 1 + n_out) % n_out];

            bool cur_inside = Cross2D(edge_vec, cur - e1) >= 0;
            bool prev_inside = Cross2D(edge_vec, prev - e1) >= 0;

            if (cur_inside) {
                if (!prev_inside)
                    new_out.push_back(LineIntersection(prev, cur, e1, e2));
                new_out.push_back(cur);
            } else if (prev_inside) {
                new_out.push_back(LineIntersection(prev, cur, e1, e2));
            }
        }
        output = new_out;
    }
    return output;
}

static float PolygonArea(const std::vector<cv::Point2f>& pts)
{
    if (pts.size() < 3) return 0.0f;
    float area = 0;
    int n = (int)pts.size();
    for (int i = 0; i < n; ++i)
        area += Cross2D(pts[i], pts[(i + 1) % n]);
    return 0.5f * std::abs(area);
}

static float RotatedRectIoU(const float* box_a, const float* box_b)
{
    // box: [x, y, w, l, r]
    float area_a = box_a[2] * box_a[3];
    float area_b = box_b[2] * box_b[3];

    float half_diag_a = std::sqrt((box_a[2]*0.5f)*(box_a[2]*0.5f) +
                                   (box_a[3]*0.5f)*(box_a[3]*0.5f));
    float half_diag_b = std::sqrt((box_b[2]*0.5f)*(box_b[2]*0.5f) +
                                   (box_b[3]*0.5f)*(box_b[3]*0.5f));
    float dx = box_a[0] - box_b[0];
    float dy = box_a[1] - box_b[1];
    if (std::sqrt(dx*dx + dy*dy) > half_diag_a + half_diag_b)
        return 0.0f;

    auto corners_a_mat = GetBEVCorners(box_a[0], box_a[1], box_a[2], box_a[3], box_a[4]);
    auto corners_b_mat = GetBEVCorners(box_b[0], box_b[1], box_b[2], box_b[3], box_b[4]);

    std::vector<cv::Point2f> subj(4), clip(4);
    for (int i = 0; i < 4; ++i) {
        subj[i] = cv::Point2f(corners_a_mat.at<float>(i, 0), corners_a_mat.at<float>(i, 1));
        clip[i] = cv::Point2f(corners_b_mat.at<float>(i, 0), corners_b_mat.at<float>(i, 1));
    }

    auto inter = PolygonIntersection(subj, clip);
    float inter_area = PolygonArea(inter);
    return inter_area / (area_a + area_b - inter_area + 1e-8f);
}

// ================================================================
// Rotate NMS CPU
// ================================================================

std::vector<int> RotateNMSCPU(const cv::Mat& boxes_bev,
                              const std::vector<float>& scores, float thresh)
{
    int n = (int)scores.size();
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<bool> suppressed(n, false);
    std::vector<int> keep;

    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (int j = i + 1; j < n; ++j) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            float iou = RotatedRectIoU(
                boxes_bev.ptr<float>(idx), boxes_bev.ptr<float>(jdx));
            if (iou > thresh)
                suppressed[jdx] = true;
        }
    }
    return keep;
}

// ================================================================
// Multi-class NMS
// ================================================================

NMSResult MultiClassScaleNMS(const cv::Mat& all_boxes,
                              const cv::Mat& all_scores,
                              const std::vector<int>& all_dir_scores)
{
    int N = all_boxes.rows;
    NMSResult result;
    result.boxes = cv::Mat(0, 9, CV_32F);  // ensure push_back works

    for (int cls_id = 0; cls_id < NUM_CLASSES; ++cls_id) {
        // Find boxes with score > NMS_SCORE_THR for this class
        std::vector<int> cls_indices;
        std::vector<float> cls_scores;
        for (int i = 0; i < N; ++i) {
            float s = all_scores.at<float>(i, cls_id);
            if (s > NMS_SCORE_THR) {
                cls_indices.push_back(i);
                cls_scores.push_back(s);
            }
        }
        if (cls_indices.empty()) continue;

        // nms_pre
        if ((int)cls_indices.size() > NMS_PRE) {
            std::vector<int> sort_idx(cls_indices.size());
            std::iota(sort_idx.begin(), sort_idx.end(), 0);
            std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
                return cls_scores[a] > cls_scores[b];
            });
            std::vector<int> top_indices;
            std::vector<float> top_scores;
            for (int i = 0; i < NMS_PRE; ++i) {
                top_indices.push_back(cls_indices[sort_idx[i]]);
                top_scores.push_back(cls_scores[sort_idx[i]]);
            }
            cls_indices = top_indices;
            cls_scores = top_scores;
        }

        std::vector<int> selected;
        int n_cls = (int)cls_indices.size();

        if (NMS_TYPE_LIST[cls_id] == "circle") {
            cv::Mat dets(n_cls, 3, CV_32F);
            for (int i = 0; i < n_cls; ++i) {
                const float* b = all_boxes.ptr<float>(cls_indices[i]);
                dets.at<float>(i, 0) = b[0];  // cx
                dets.at<float>(i, 1) = b[1];  // cy
                dets.at<float>(i, 2) = cls_scores[i];
            }
            auto sel = CircleNMS(dets, NMS_RADIUS_THR_LIST[cls_id]);
            for (int s : sel) selected.push_back(cls_indices[s]);
        } else {
            float rescale = NMS_RESCALE_FACTOR[cls_id];
            cv::Mat boxes_bev(n_cls, 5, CV_32F);
            for (int i = 0; i < n_cls; ++i) {
                const float* b = all_boxes.ptr<float>(cls_indices[i]);
                float* row = boxes_bev.ptr<float>(i);
                row[0] = b[0];                 // x
                row[1] = b[1];                 // y
                row[2] = b[3] * rescale;       // w (note: boxes=[x,y,z,w,l,h,r,...])
                row[3] = b[4] * rescale;       // l
                row[4] = b[6];                 // r
            }
            auto sel = RotateNMSCPU(boxes_bev, cls_scores, NMS_THR_LIST[cls_id]);
            for (int s : sel) selected.push_back(cls_indices[s]);
        }

        for (int k : selected) {
            result.boxes.push_back(all_boxes.row(k));
            result.scores.push_back(all_scores.at<float>(k, cls_id));
            result.labels.push_back(cls_id);
            result.dir_scores.push_back(all_dir_scores[k]);
        }
    }

    // max_num
    int total = (int)result.scores.size();
    if (total > MAX_NUM) {
        std::vector<int> order(total);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return result.scores[a] > result.scores[b];
        });

        cv::Mat new_boxes(MAX_NUM, 9, CV_32F);
        std::vector<float> new_scores(MAX_NUM);
        std::vector<int> new_labels(MAX_NUM);
        std::vector<int> new_dirs(MAX_NUM);
        for (int i = 0; i < MAX_NUM; ++i) {
            int o = order[i];
            result.boxes.row(o).copyTo(new_boxes.row(i));
            new_scores[i] = result.scores[o];
            new_labels[i] = result.labels[o];
            new_dirs[i] = result.dir_scores[o];
        }
        result.boxes = new_boxes;
        result.scores = new_scores;
        result.labels = new_labels;
        result.dir_scores = new_dirs;
    }

    return result;
}

// ================================================================
// Direction correction
// ================================================================

static float LimitPeriod(float val, float offset, float period)
{
    return val - std::floor(val / period + offset) * period;
}

void CorrectYaw(cv::Mat& boxes, const std::vector<int>& dir_scores)
{
    int N = boxes.rows;
    for (int i = 0; i < N; ++i) {
        float yaw = boxes.at<float>(i, 6);
        float dir_rot = LimitPeriod(yaw - DIR_OFFSET, 0.0f, CV_PI);
        boxes.at<float>(i, 6) = dir_rot + DIR_OFFSET + CV_PI * dir_scores[i];
    }
}

}  // namespace postprocess
