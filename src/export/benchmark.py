"""Latency + size benchmark: PyTorch fp32 vs ONNX fp32 vs ONNX INT8.

Produces the table you put in the README / resume:
    "Cut inference latency X% and model size Y% via ONNX + INT8 quantization."
"""
import os
import time

from src.core.config import Config
from src.core.logging import logger


def _file_mb(path):
    return round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else None


def _save(rows):
    import json
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(Config.ARTIFACT_DIR, "benchmark.json"), "w") as f:
        json.dump(rows, f, indent=2)


def _time_callable(fn, runs=50, warmup=10):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return round((time.perf_counter() - start) / runs * 1000, 2)  # ms/inference


def benchmark(model, img_size: int = None, runs: int = 50):
    import numpy as np
    import torch

    img_size = img_size or Config.IMG_SIZE
    x_np = np.random.randn(1, 3, img_size, img_size).astype("float32")
    x_torch = torch.from_numpy(x_np)

    rows = []
    from src.export.onnx_export import export_onnx, quantize_int8, load_onnx_session

    # 1) PyTorch fp32
    try:
        model.eval()
        with torch.no_grad():
            rows.append({"engine": "pytorch_fp32",
                         "latency_ms": _time_callable(lambda: model(x_torch), runs),
                         "size_mb": _file_mb(Config.CHECKPOINT_PATH)})
        logger.info("Benchmarked pytorch_fp32"); _save(rows)
    except Exception as exc:
        logger.warning(f"PyTorch benchmark skipped: {exc}")

    # 2) ONNX fp32
    try:
        export_onnx(model, img_size=img_size)
        sess = load_onnx_session(Config.ONNX_PATH)
        iname = sess.get_inputs()[0].name
        rows.append({"engine": "onnx_fp32",
                     "latency_ms": _time_callable(lambda: sess.run(None, {iname: x_np}), runs),
                     "size_mb": _file_mb(Config.ONNX_PATH)})
        logger.info("Benchmarked onnx_fp32"); _save(rows)
    except Exception as exc:
        logger.warning(f"ONNX fp32 benchmark skipped: {exc}")

    # 3) ONNX INT8
    try:
        quantize_int8()
        sess8 = load_onnx_session(Config.ONNX_INT8_PATH)
        iname8 = sess8.get_inputs()[0].name
        rows.append({"engine": "onnx_int8",
                     "latency_ms": _time_callable(lambda: sess8.run(None, {iname8: x_np}), runs),
                     "size_mb": _file_mb(Config.ONNX_INT8_PATH)})
        logger.info("Benchmarked onnx_int8"); _save(rows)
    except Exception as exc:
        logger.warning(f"INT8 quantization skipped: {exc}")

    _report(rows)
    return rows


def _report(rows):
    import sys
    base = next((r for r in rows if r["engine"] == "pytorch_fp32"), rows[0])
    print("\n=== Inference benchmark (1 image, CPU) ===")
    print(f"{'engine':16} {'latency_ms':>12} {'size_mb':>10} {'speedup':>9}")
    for r in rows:
        speed = f"{base['latency_ms'] / r['latency_ms']:.2f}x" if r["latency_ms"] else "-"
        print(f"{r['engine']:16} {r['latency_ms']:>12} {str(r['size_mb']):>10} {speed:>9}")
    import json
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(Config.ARTIFACT_DIR, "benchmark.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {Config.ARTIFACT_DIR}/benchmark.json")
    sys.stdout.flush()
