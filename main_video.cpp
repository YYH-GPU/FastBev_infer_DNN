#include "fastbev_infer.h"
#include "visualize.h"
#include "loader/nuscenes_loader.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <set>
#include <cstdio>

namespace fs = std::filesystem;

// ------------------------------------------------------------------
// Collect all samples for a scene, sorted by timestamp, skipping
// those without enough temporal history.
// ------------------------------------------------------------------
static std::vector<std::string> GetSceneSamples(const std::string& scene_token,
                                                 int n_sweeps,
                                                 const nuscenes::Tables& tables)
{
    std::vector<std::pair<int64_t, std::string>> pairs;
    for (const auto& s : tables.sample_list) {
        if (s["scene_token"].get<std::string>() == scene_token)
            pairs.push_back({s["timestamp"].get<int64_t>(), s["token"].get<std::string>()});
    }
    std::sort(pairs.begin(), pairs.end());

    // Filter: only keep samples with enough temporal history
    std::vector<std::string> tokens;
    tokens.reserve(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        bool ok = true;
        std::string tok = pairs[i].second;
        for (int s = 1; s < n_sweeps && ok; ++s) {
            auto it = tables.sample_by_token.find(tok);
            if (it == tables.sample_by_token.end()) { ok = false; break; }
            const auto& js = it->second;
            if (!js.contains("prev") || js["prev"].is_null() ||
                js["prev"].get<std::string>().empty()) {
                ok = false;
                break;
            }
            tok = js["prev"].get<std::string>();
        }
        if (ok)
            tokens.push_back(pairs[i].second);
    }
    return tokens;
}

static void PrintUsage(const char* prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Scene video mode — runs inference on all samples of a scene\n");
    printf("and stitches BEV frames into a video.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --scene TOKEN        NuScenes scene token (required)\n");
    printf("  --no-gpu             Use CPU inference\n");
    printf("  --no-export          Skip image export (if images already exist)\n");
    printf("  --fps N              Video frame rate (default: 10)\n");
    printf("  --out-video PATH     Output video path (default: output/bev_scene.mp4)\n");
    printf("  --score-thr FLOAT    Score threshold (default: 0.15)\n");
    printf("  --list-scenes        List all scene tokens and exit\n");
    printf("  --list-tokens [N]    List sample tokens grouped by scene (default: 50)\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s --list-scenes\n", prog);
    printf("  %s --scene 3e8750f331d7499e9b5123e9eb70f2e2\n", prog);
    printf("  %s --scene 3e8750f331d7499e9b5123e9eb70f2e2 --no-gpu --fps 5\n", prog);
}

int main(int argc, char* argv[])
{
    // ---- Parse args ----
    std::string scene_token;
    std::string model_2d     = "fastbev_2d.onnx";
    std::string model_3d     = "fastbev_3d.onnx";
    std::string nuscenes_root = "data";
    std::string dataroot     = "data";
    std::string image_dir    = "images";
    std::string out_dir      = "output";
    std::string out_video    = "output/bev_scene.mp4";
    float score_thr          = 0.15f;
    int   fps                = 10;
    bool  use_gpu            = true;
    bool  no_export          = false;
    bool  list_scenes        = false;
    bool  list_tokens        = false;
    int   list_limit         = 50;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc) {
            scene_token = argv[++i];
        } else if (a == "--no-gpu") {
            use_gpu = false;
        } else if (a == "--no-export") {
            no_export = true;
        } else if (a == "--fps" && i + 1 < argc) {
            fps = std::stoi(argv[++i]);
        } else if (a == "--out-video" && i + 1 < argc) {
            out_video = argv[++i];
        } else if (a == "--score-thr" && i + 1 < argc) {
            score_thr = std::stof(argv[++i]);
        } else if (a == "--list-scenes") {
            list_scenes = true;
        } else if (a == "--list-tokens") {
            list_tokens = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                try { list_limit = std::stoi(argv[i + 1]); ++i; }
                catch (...) {}
            }
        } else if (a == "--model-2d" && i + 1 < argc) {
            model_2d = argv[++i];
        } else if (a == "--model-3d" && i + 1 < argc) {
            model_3d = argv[++i];
        } else if (a == "--nuscenes-root" && i + 1 < argc) {
            nuscenes_root = argv[++i];
        } else if (a == "--dataroot" && i + 1 < argc) {
            dataroot = argv[++i];
        } else if (a == "--image-dir" && i + 1 < argc) {
            image_dir = argv[++i];
        } else if (a == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
            out_video = out_dir + "/bev_scene.mp4";
        } else if (a == "--help" || a == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    // ---- Load tables ----
    printf("Loading NuScenes from: %s ...\n", nuscenes_root.c_str());
    auto tables = nuscenes::LoadTables(nuscenes_root);

    // ---- List scenes ----
    if (list_scenes) {
        printf("\n=== Available Scenes (v1.0-mini) ===\n");
        // Gather unique scene tokens from samples
        std::set<std::string> scenes;
        std::unordered_map<std::string, int> sample_counts;
        for (const auto& s : tables.sample_list) {
            std::string sc = s["scene_token"].get<std::string>();
            scenes.insert(sc);
            sample_counts[sc]++;
        }
        int idx = 0;
        for (const auto& sc : scenes) {
            int ns = GetSceneSamples(sc, nuscenes::N_SWEEPS, tables).size();
            printf("  [%d] %s  (%d samples, %d processable)\n",
                   idx++, sc.c_str(), sample_counts[sc], ns);
        }
        printf("\nExample: %s --scene <TOKEN>\n", argv[0]);
        return 0;
    }

    // ---- List tokens ----
    if (list_tokens) {
        // Group sample tokens by scene
        std::unordered_map<std::string, std::vector<std::string>> scene_samples;
        for (const auto& s : tables.sample_list) {
            std::string sc = s["scene_token"].get<std::string>();
            std::string tok = s["token"].get<std::string>();
            scene_samples[sc].push_back(tok);
        }

        // Collect unique scenes
        std::vector<std::string> scenes;
        for (const auto& s : tables.sample_list) {
            std::string sc = s["scene_token"].get<std::string>();
            if (std::find(scenes.begin(), scenes.end(), sc) == scenes.end())
                scenes.push_back(sc);
        }

        printf("\n=== Available Sample Tokens (v1.0-mini) ===\n");
        int total_shown = 0;
        for (size_t si = 0; si < scenes.size() && total_shown < list_limit; ++si) {
            auto& toks = scene_samples[scenes[si]];
            printf("\nScene: %s  (%zu samples)\n", scenes[si].c_str(), toks.size());
            int n_show = std::min((int)toks.size(), list_limit - total_shown);
            for (int i = 0; i < n_show; ++i) {
                printf("  [%d] %s\n", total_shown, toks[i].c_str());
                ++total_shown;
            }
            if (total_shown >= list_limit && si < scenes.size() - 1)
                printf("  ... (use --list-tokens N to see more)\n");
        }
        printf("\nExample: %s --scene <SCENE_TOKEN>\n", argv[0]);
        return 0;
    }

    if (scene_token.empty()) {
        printf("Error: --scene is required. Use --list-scenes to see available scenes.\n");
        return 1;
    }

    // ---- Collect samples ----
    auto sample_tokens = GetSceneSamples(scene_token, nuscenes::N_SWEEPS, tables);
    if (sample_tokens.empty()) {
        printf("Error: no processable samples in this scene (need %d temporal sweeps).\n",
               nuscenes::N_SWEEPS);
        return 1;
    }
    printf("Scene: %s\n", scene_token.c_str());
    printf("Processable samples: %zu\n", sample_tokens.size());

    // ---- Init inference engine ----
    FastBEVInfer::Config cfg;
    cfg.model_2d_path = model_2d;
    cfg.model_3d_path = model_3d;
    cfg.nuscenes_root = nuscenes_root;
    cfg.dataroot = dataroot;
    cfg.image_dir = image_dir;
    cfg.out_dir = out_dir;
    cfg.score_thr = score_thr;
    cfg.use_gpu = use_gpu;
    cfg.no_export = no_export;

    printf("Initializing FastBEV...  GPU: %s\n", use_gpu ? "enabled" : "disabled");
    FastBEVInfer infer(cfg);

    // ---- Process all samples, collect BEV frames ----
    fs::create_directories(out_dir);
    std::vector<cv::Mat> frames;
    frames.reserve(sample_tokens.size());
    int total = (int)sample_tokens.size();

    printf("\n=== Processing %d samples ===\n", total);

    for (int i = 0; i < total; ++i) {
        printf("\n--- Sample %d/%d: %s ---\n", i + 1, total, sample_tokens[i].c_str());

        try {
            auto detections = infer.Run(sample_tokens[i]);

            // Load current-sweep camera images
            std::vector<cv::Mat> cam_imgs;
            std::vector<std::string> cam_names;
            for (const auto& cam : nuscenes::CAMERA_ORDER) {
                for (const auto& ext : {".png", ".jpg", ".jpeg"}) {
                    std::string path = image_dir + "/" + cam + "_sweep0" + ext;
                    cv::Mat img = cv::imread(path);
                    if (!img.empty()) {
                        cam_imgs.push_back(img);
                        cam_names.push_back(cam);
                        break;
                    }
                }
            }
            printf("  Loaded %zu camera images from %s\n", cam_imgs.size(), image_dir.c_str());
            if (cam_imgs.empty()) {
                printf("  WARNING: no camera images found! Listing directory:\n");
                for (const auto& entry : fs::directory_iterator(image_dir)) {
                    printf("    %s\n", entry.path().filename().string().c_str());
                }
            }

            // Combined frame: cameras + BEV
            cv::Mat frame = visualize::DrawCombinedFrame(
                detections, cam_imgs, cam_names,
                infer.GetLastCalib(), score_thr);

            // Progress overlay
            char info[64];
            snprintf(info, sizeof(info), "Frame %d / %d", i + 1, total);
            cv::putText(frame, info, cv::Point(1700, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            auto ts_us = tables.sample_by_token[sample_tokens[i]]["timestamp"].get<int64_t>();
            char ts_str[64];
            snprintf(ts_str, sizeof(ts_str), "t=%.2fs", ts_us / 1e6);
            cv::putText(frame, ts_str, cv::Point(1700, 45),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

            frames.push_back(frame);
            printf("  -> %d detections, frame captured\n", (int)detections.size());
        } catch (const std::exception& e) {
            printf("  ERROR: %s — skipping this sample\n", e.what());
        }
    }

    if (frames.empty()) {
        printf("Error: no frames collected.\n");
        return 1;
    }

    // ---- Write video ----
    printf("\nWriting video: %s (%zu frames, %d fps)\n",
           out_video.c_str(), frames.size(), fps);
    cv::VideoWriter writer(out_video,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps,
                           cv::Size(1800, 1200));
    if (!writer.isOpened()) {
        printf("ERROR: Failed to open video writer. Try different codec.\n");
        // Fallback: try MJPG codec
        cv::VideoWriter writer2(out_video + ".avi",
                                cv::VideoWriter::fourcc('M','J','P','G'),
                                fps,
                                cv::Size(1800, 1200));
        if (writer2.isOpened()) {
            for (auto& f : frames) writer2.write(f);
            writer2.release();
            printf("Video saved as %s.avi\n", out_video.c_str());
            return 0;
        }
        // Last resort: save frames as individual PNGs
        printf("Saving frames as PNGs instead...\n");
        for (size_t i = 0; i < frames.size(); ++i) {
            char p[256];
            snprintf(p, sizeof(p), "%s/frame_%04zu.png", out_dir.c_str(), i);
            cv::imwrite(p, frames[i]);
        }
        printf("Frames saved to %s/\n", out_dir.c_str());
    } else {
        for (auto& f : frames) writer.write(f);
        writer.release();
        printf("Video saved: %s\n", out_video.c_str());
    }

    printf("\nDone. %d frames → %s\n", (int)frames.size(), out_video.c_str());
    return 0;
}
