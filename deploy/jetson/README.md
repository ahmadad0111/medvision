# MedVision on NVIDIA Jetson (edge inference)

Run the trained classifier on a Jetson device using **ONNX Runtime** (or
**TensorRT**) — no PyTorch on the edge. You train and export on a workstation,
then deploy a small inference image to the Jetson.

```
Workstation:  train (PyTorch)  →  export ONNX + INT8 (scripts.export)
Jetson:       ONNX Runtime / TensorRT  →  FastAPI /predict + web UI
```

## Prerequisites

- A Jetson with JetPack flashed (Nano → JetPack 4.6; Orin → JetPack 5/6).
- `model.onnx` in `artifacts/` (produced by `python -m scripts.export`). Copy it to the Jetson.
- Docker with the NVIDIA runtime (default on JetPack).

## Option A — ONNX Runtime (simplest)

```bash
# on the Jetson, from the repo root (with artifacts/model.onnx present)
docker build -f deploy/jetson/Dockerfile -t medvision-edge .
docker run --runtime nvidia -p 8000:8000 medvision-edge
# open http://<jetson-ip>:8000/app  or POST an image to /predict
```

> Match the base image in `deploy/jetson/Dockerfile` to your JetPack/L4T version.
> If ONNX Runtime rejects the model's opset, re-export with a lower opset
> (`opset_version=13` in `src/export/onnx_export.py`).

## Option B — TensorRT (fastest)

```bash
# on the Jetson (trtexec ships with JetPack)
bash deploy/jetson/convert_tensorrt.sh artifacts/model.onnx artifacts/model.trt
trtexec --loadEngine=artifacts/model.trt --iterations=100   # benchmark
```

FP16 on the Jetson GPU typically gives the best latency/Watt. (Wiring the `.trt`
engine into the API is a small follow-up; the ONNX path above works out of the box.)

## Benchmark (fill in from your device)

| Device / runtime | Latency (ms/image) | Notes |
|---|---|---|
| Jetson Nano — ONNX Runtime (CUDA) | _…_ | 4 GB, Maxwell GPU |
| Jetson Nano — TensorRT FP16 | _…_ | fastest |
| (reference) x86 CPU — ONNX fp32 | 33 | from `scripts.export` |

## Notes

- This edge path returns **prediction + top-k probabilities**. Grad-CAM needs
  gradients (PyTorch) and stays on the full server (`src.api.main`).
- The existing web UI (`frontend/`) works against the edge API — the heatmap
  panel simply stays empty on edge.
- The original **Jetson Nano (4 GB)** is ideal for this vision model; it is *not*
  suitable for the InsightDocs LLM/RAG stack — use a cloud LLM or a Jetson Orin
  for that.
