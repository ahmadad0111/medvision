"""Lightweight ONNX Runtime predictor for edge devices (e.g., Jetson Nano).

No PyTorch/timm at inference — only ONNX Runtime + NumPy + Pillow — so it runs
in a small footprint. On Jetson it uses the TensorRT / CUDA execution providers
when available, falling back to CPU elsewhere. (Grad-CAM needs gradients and
stays on the full PyTorch server; this path is prediction + probabilities.)
"""
import numpy as np

from src.core.config import Config
from src.core.logging import logger


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _default_providers():
    import onnxruntime as ort
    available = ort.get_available_providers()
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    chosen = [p for p in preferred if p in available]
    return chosen or ["CPUExecutionProvider"]


class ONNXPredictor:
    def __init__(self, onnx_path: str = None, providers=None):
        import onnxruntime as ort
        self.onnx_path = onnx_path or Config.ONNX_PATH
        self.session = ort.InferenceSession(self.onnx_path,
                                            providers=providers or _default_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array(Config.MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(Config.STD, dtype=np.float32).reshape(3, 1, 1)
        logger.info(f"ONNX predictor ready ({self.onnx_path}) "
                    f"providers={self.session.get_providers()}")

    def _preprocess(self, pil_image):
        img = pil_image.convert("RGB").resize((Config.IMG_SIZE, Config.IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0     # HWC in [0,1]
        arr = arr.transpose(2, 0, 1)                        # CHW
        arr = (arr - self.mean) / self.std                  # normalise
        return arr[None, ...].astype(np.float32)            # NCHW

    def predict(self, pil_image, top_k: int = 3):
        from src.inference.predictor import softmax_topk
        x = self._preprocess(pil_image)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return {
            "prediction": {"class_id": idx, "label": Config.labels()[idx],
                           "prob": round(float(probs[idx]), 4)},
            "top_k": softmax_topk([float(p) for p in probs], k=top_k),
        }
