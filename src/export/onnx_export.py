"""Export the trained model to ONNX and (optionally) quantize to INT8."""
import os

from src.core.config import Config
from src.core.logging import logger


def export_onnx(model, onnx_path: str = None, img_size: int = None):
    import torch
    onnx_path = onnx_path or Config.ONNX_PATH
    img_size = img_size or Config.IMG_SIZE
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    logger.info(f"Exported ONNX -> {onnx_path}")
    return onnx_path


def quantize_int8(onnx_path: str = None, int8_path: str = None):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    onnx_path = onnx_path or Config.ONNX_PATH
    int8_path = int8_path or Config.ONNX_INT8_PATH
    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
    logger.info(f"Quantized INT8 ONNX -> {int8_path}")
    return int8_path


def load_onnx_session(onnx_path: str):
    import onnxruntime as ort
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
