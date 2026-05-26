#!/bin/bash
# Run FastBEV video inference (build first with build.sh)
# Usage:
#   bash run_video.sh                                    # default scene, GPU
#   bash run_video.sh --list-scenes                      # list scenes, exit
#   bash run_video.sh --list-tokens                      # list sample tokens
#   bash run_video.sh --list-tokens 20                   # list first 20 tokens
#   bash run_video.sh <SCENE_TOKEN>                      # custom scene, GPU
#   bash run_video.sh <SCENE_TOKEN> --no-gpu             # custom scene, CPU
#   bash run_video.sh <SCENE_TOKEN> --no-gpu --fps 5     # custom scene, CPU, 5 fps
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_2D="$SCRIPT_DIR/models/fastbev_2d.onnx"
MODEL_3D="$SCRIPT_DIR/models/fastbev_3d.onnx"
NUSCENES_ROOT="$SCRIPT_DIR/data"
DATAROOT="$SCRIPT_DIR/data"
IMAGE_DIR="$SCRIPT_DIR/images"
OUT_DIR="$SCRIPT_DIR/output"

# Parse first positional arg as scene token if it doesn't start with --
SCENE="cc8c0bf57f984915a77078b10eb33198"
EXTRA_ARGS=()
if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then
    SCENE="$1"
    shift
fi
EXTRA_ARGS=("$@")

# If --list-* mode, pass through without requiring scene
LIST_MODE=0
for a in "${EXTRA_ARGS[@]}"; do
    case "$a" in
        --list-scenes|--list-tokens) LIST_MODE=1 ;;
    esac
done

if [ "$LIST_MODE" -eq 0 ] && [ -z "$SCENE" ] && ! echo "${EXTRA_ARGS[@]}" | grep -q '\--scene'; then
    echo "No scene specified. Use --list-scenes or --list-tokens to see available tokens."
    echo "Usage: bash run_video.sh [SCENE_TOKEN] [--list-scenes|--list-tokens [N]] [--no-gpu] ..."
    echo "Run with --help for full options."
    exit 1
fi

echo "=== Running video inference ==="
[ -n "$SCENE" ] && echo "Scene:     $SCENE"
echo "Model 2D:  $MODEL_2D"
echo "Model 3D:  $MODEL_3D"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "Extra args: ${EXTRA_ARGS[*]}"

CMD=("./build/fastbev_video"
    --model-2d "$MODEL_2D"
    --model-3d "$MODEL_3D"
    --nuscenes-root "$NUSCENES_ROOT"
    --dataroot "$DATAROOT"
    --image-dir "$IMAGE_DIR"
    --out-dir "$OUT_DIR"
    --fps 10
)
[ -n "$SCENE" ] && CMD+=(--scene "$SCENE")
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"

echo ""
echo "=== Output in: $OUT_DIR ==="
ls -la "$OUT_DIR" 2>/dev/null
