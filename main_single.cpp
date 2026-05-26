#include "fastbev_infer.h"
#include "loader/nuscenes_loader.h"
#include <iostream>
#include <string>
#include <iomanip>
#include <set>
#include <algorithm>

static void PrintUsage(const char* prog)
{
    printf("Usage:\n");
    printf("  %s --token SAMPLE_TOKEN [options]   Run inference on a single sample\n", prog);
    printf("  %s --list-tokens [N]                List available sample tokens\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --token TOKEN         NuScenes sample token to run inference on\n");
    printf("  --list-tokens [N]     List sample tokens (optionally limit to N, default 50)\n");
    printf("  --model-2d PATH       2D ONNX model path (default: models/fastbev_2d.onnx)\n");
    printf("  --model-3d PATH       3D ONNX model path (default: models/fastbev_3d.onnx)\n");
    printf("  --nuscenes-root DIR   NuScenes v1.0-mini parent dir (default: data)\n");
    printf("  --dataroot DIR        NuScenes images root (default: data)\n");
    printf("  --image-dir DIR       Working dir for exported images (default: images)\n");
    printf("  --out-dir DIR         Output dir for visualizations (default: output)\n");
    printf("  --score-thr FLOAT     Score threshold (default: 0.3)\n");
    printf("  --no-gpu              Disable GPU inference\n");
    printf("  --no-export           Skip image export (if already done)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --list-tokens\n", prog);
    printf("  %s --list-tokens 20\n", prog);
    printf("  %s --token 6402fd1ffaf041d0b9162bd92a7ba0a2\n", prog);
    printf("  %s --token TOKEN --score-thr 0.1 --no-gpu\n", prog);
}

int main(int argc, char* argv[])
{
    // Parse arguments
    std::vector<std::string> tokens;
    std::string model_2d = "fastbev_2d.onnx";
    std::string model_3d = "fastbev_3d.onnx";
    std::string nuscenes_root = "data";
    std::string dataroot = "data";
    std::string image_dir = "images";
    std::string out_dir = "output";
    float score_thr = 0.3f;
    bool use_gpu = true;
    bool no_export = false;
    bool list_tokens = false;
    int list_limit = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--token" && i + 1 < argc) {
            tokens.push_back(argv[++i]);
        } else if (arg == "--model-2d" && i + 1 < argc) {
            model_2d = argv[++i];
        } else if (arg == "--model-3d" && i + 1 < argc) {
            model_3d = argv[++i];
        } else if (arg == "--nuscenes-root" && i + 1 < argc) {
            nuscenes_root = argv[++i];
        } else if (arg == "--dataroot" && i + 1 < argc) {
            dataroot = argv[++i];
        } else if (arg == "--image-dir" && i + 1 < argc) {
            image_dir = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--score-thr" && i + 1 < argc) {
            score_thr = std::stof(argv[++i]);
        } else if (arg == "--no-gpu") {
            use_gpu = false;
        } else if (arg == "--no-export") {
            no_export = true;
        } else if (arg == "--list-tokens") {
            list_tokens = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                try { list_limit = std::stoi(argv[i + 1]); ++i; }
                catch (...) {}
            }
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    // ---- List tokens ----
    if (list_tokens) {
        printf("Loading NuScenes from: %s ...\n", nuscenes_root.c_str());
        auto tables = nuscenes::LoadTables(nuscenes_root);

        // Group sample tokens by scene
        std::unordered_map<std::string, std::vector<std::string>> scene_samples;
        for (const auto& s : tables.sample_list) {
            std::string sc = s["scene_token"].get<std::string>();
            std::string tok = s["token"].get<std::string>();
            scene_samples[sc].push_back(tok);
        }

        // Collect all (scene_token, first_sample) pairs sorted by timestamp
        std::vector<std::pair<std::string, std::string>> scenes;
        for (const auto& s : tables.sample_list) {
            std::string sc = s["scene_token"].get<std::string>();
            if (std::find_if(scenes.begin(), scenes.end(),
                             [&](auto& p) { return p.first == sc; }) == scenes.end())
                scenes.push_back({sc, s["token"].get<std::string>()});
        }

        printf("\n=== Available Sample Tokens (v1.0-mini) ===\n");
        int total_shown = 0;
        for (size_t si = 0; si < scenes.size() && total_shown < list_limit; ++si) {
            const auto& sc = scenes[si].first;
            auto& toks = scene_samples[sc];
            printf("\nScene: %s  (%zu samples)\n", sc.c_str(), toks.size());
            int n_show = std::min((int)toks.size(), list_limit - total_shown);
            for (int i = 0; i < n_show; ++i) {
                printf("  [%d] %s\n", total_shown, toks[i].c_str());
                ++total_shown;
            }
            if (total_shown >= list_limit && si < scenes.size() - 1)
                printf("  ... (use --list-tokens N to see more)\n");
        }
        printf("\nExample: %s --token <TOKEN>\n", argv[0]);
        return 0;
    }

    if (tokens.empty()) {
        printf("Error: --token is required. Use --list-tokens to see available tokens.\n\n");
        PrintUsage(argv[0]);
        return 1;
    }

    // Configure inference
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

    // Initialize the inference engine (models loaded once)
    printf("Initializing FastBEV inference engine...\n");
    printf("  model_2d: %s\n", cfg.model_2d_path.c_str());
    printf("  model_3d: %s\n", cfg.model_3d_path.c_str());
    printf("  nuscenes_root: %s\n", cfg.nuscenes_root.c_str());

    try {
        FastBEVInfer infer(cfg);

        // Run inference on each token
        for (const auto& token : tokens) {
            printf("\n========================================\n");
            printf("Processing token: %s\n", token.c_str());
            printf("========================================\n");

            auto detections = infer.Run(token);
            infer.Visualize(detections);
        }

        printf("\nAll done.\n");
    } catch (const std::exception& e) {
        printf("FATAL ERROR: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("FATAL ERROR: unknown exception\n");
        return 1;
    }
    return 0;
}
