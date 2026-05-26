#!/bin/bash
# Run FastBEV single-image inference (build first with build.sh)
# Usage:
#   bash run_image.sh                                    # default token, GPU
#   bash run_image.sh --list-tokens                      # list available tokens
#   bash run_image.sh --list-tokens 20                   # list first 20 tokens
#   bash run_image.sh <TOKEN>                            # custom token, GPU
#   bash run_image.sh <TOKEN> --no-gpu                   # custom token, CPU
#   bash run_image.sh --token <TOKEN> --score-thr 0.1    # named args
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_2D="$SCRIPT_DIR/models/fastbev_2d.onnx"
MODEL_3D="$SCRIPT_DIR/models/fastbev_3d.onnx"
NUSCENES_ROOT="$SCRIPT_DIR/data"
DATAROOT="$SCRIPT_DIR/data"
IMAGE_DIR="$SCRIPT_DIR/images"
OUT_DIR="$SCRIPT_DIR/output"

# Parse first positional arg as token if it doesn't start with --
TOKEN="6402fd1ffaf041d0b9162bd92a7ba0a2"
EXTRA_ARGS=()
if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then
    TOKEN="$1"
    shift
fi
EXTRA_ARGS=("$@")

# If --list-tokens, pass through without requiring token
LIST_MODE=0
for a in "${EXTRA_ARGS[@]}"; do
    [[ "$a" == "--list-tokens" ]] && LIST_MODE=1
done

if [ "$LIST_MODE" -eq 0 ] && [ -z "$TOKEN" ] && ! echo "${EXTRA_ARGS[@]}" | grep -q '\--token'; then
    # No token provided — show usage
    echo "No token specified. Use --list-tokens to see available tokens."
    echo "Usage: bash run_image.sh [TOKEN] [--list-tokens [N]] [--no-gpu] ..."
    echo "Run with --help for full options."
    exit 1
fi

echo "=== Running image inference ==="
[ -n "$TOKEN" ] && echo "Token: $TOKEN"
echo "Model 2D:  $MODEL_2D"
echo "Model 3D:  $MODEL_3D"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "Extra args: ${EXTRA_ARGS[*]}"

CMD=("./build/fastbev_infer"
    --model-2d "$MODEL_2D"
    --model-3d "$MODEL_3D"
    --nuscenes-root "$NUSCENES_ROOT"
    --dataroot "$DATAROOT"
    --image-dir "$IMAGE_DIR"
    --out-dir "$OUT_DIR"
    --score-thr 0.3
)
[ -n "$TOKEN" ] && CMD+=(--token "$TOKEN")
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"

echo ""
echo "=== Output in: $OUT_DIR ==="
ls -la "$OUT_DIR" 2>/dev/null
