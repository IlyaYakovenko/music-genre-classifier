#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-exported_model/music_genre_classifier_ensemble.onnx}"
ENGINE_PATH="${2:-exported_model/music_genre_classifier_ensemble.plan}"
PRECISION="${3:-fp16}"

MIN_SHAPE="${MIN_SHAPE:-input:1x3x224x224}"
OPT_SHAPE="${OPT_SHAPE:-input:8x3x224x224}"
MAX_SHAPE="${MAX_SHAPE:-input:16x3x224x224}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "ERROR: trtexec was not found."
  echo "Install NVIDIA TensorRT or run this script inside an NVIDIA TensorRT Docker container."
  exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
  echo "ERROR: ONNX model was not found: $ONNX_PATH"
  echo "Run: uv run dvc pull -r models_remote exported_model.dvc"
  exit 1
fi

if [ -f "${ONNX_PATH}.data" ]; then
  echo "Found external ONNX weights: ${ONNX_PATH}.data"
else
  echo "WARNING: external ONNX data file was not found: ${ONNX_PATH}.data"
  echo "If ONNX Runtime/TensorRT cannot load the model, pull exported_model.dvc again."
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

PRECISION_FLAG=""
if [ "$PRECISION" = "fp16" ]; then
  PRECISION_FLAG="--fp16"
elif [ "$PRECISION" = "fp32" ]; then
  PRECISION_FLAG=""
else
  echo "ERROR: unsupported precision: $PRECISION"
  echo "Supported values: fp32, fp16"
  exit 1
fi

echo "Building TensorRT engine..."
echo "ONNX: $ONNX_PATH"
echo "ENGINE: $ENGINE_PATH"
echo "PRECISION: $PRECISION"
echo "MIN_SHAPE: $MIN_SHAPE"
echo "OPT_SHAPE: $OPT_SHAPE"
echo "MAX_SHAPE: $MAX_SHAPE"

trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --minShapes="$MIN_SHAPE" \
  --optShapes="$OPT_SHAPE" \
  --maxShapes="$MAX_SHAPE" \
  $PRECISION_FLAG \
  --verbose

echo "Saved TensorRT engine: $ENGINE_PATH"
