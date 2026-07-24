"""Inference: preprocess an image, predict the lesion class, and (optionally)
produce a Grad-CAM explanation overlay.
"""
import base64
import io

from src.core.config import Config
from src.core.logging import logger


def softmax_topk(probs, k: int = 3):
    labels = Config.labels()
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return [{"class_id": i, "label": labels[i], "prob": round(float(probs[i]), 4)}
            for i in ranked[:k]]


class Predictor:
    def __init__(self, checkpoint_path: str = None, device: str = None):
        from src.core.seed import resolve_device
        from src.models.model import build_model_from_checkpoint, find_gradcam_target_layer
        from src.explain.gradcam import GradCAM

        self.device = device or resolve_device(Config.DEVICE)
        path = checkpoint_path or Config.CHECKPOINT_PATH
        self.model, _ = build_model_from_checkpoint(path, device=self.device)
        self.model.to(self.device)
        self._target = find_gradcam_target_layer(self.model)
        self._gradcam_cls = GradCAM
        logger.info(f"Predictor ready (device={self.device}, checkpoint={path})")

    def _preprocess(self, pil_image):
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(Config.MEAN, Config.STD),
        ])
        return tf(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, pil_image, explain: bool = True, top_k: int = 3):
        from src.explain.gradcam import overlay_heatmap

        x = self._preprocess(pil_image)
        result = {}
        if explain and self._target is not None:
            cam_engine = self._gradcam_cls(self.model, self._target)
            cam, class_idx, probs = cam_engine(x)
            cam_engine.remove()
            overlay = overlay_heatmap(pil_image, cam)
            result["gradcam_png_base64"] = _png_b64(overlay)
        else:
            import torch
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
            class_idx = int(probs.argmax())

        result.update({
            "prediction": {"class_id": class_idx, "label": Config.labels()[class_idx],
                           "prob": round(float(probs[class_idx]), 4)},
            "top_k": softmax_topk(list(probs), k=top_k),
        })
        return result


def _png_b64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
