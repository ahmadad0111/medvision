#!/usr/bin/env bash
# Convert the exported ONNX model to a TensorRT engine for maximum Jetson speed.
# Run this ON the Jetson (trtexec ships with JetPack). FP16 roughly halves latency
# and memory vs FP32 on the Maxwell/Ampere GPU.
set -e
ONNX="${1:-artifacts/model.onnx}"
ENGINE="${2:-artifacts/model.trt}"

trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --workspace=512 \
  --explicitBatch

echo "Saved TensorRT engine -> $ENGINE"
echo "Tip: benchmark with  trtexec --loadEngine=$ENGINE --iterations=100"
