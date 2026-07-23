"""Grad-CAM — a lightweight, dependency-free implementation.

Registers forward/backward hooks on a target convolutional layer, then produces
a class-discriminative heatmap: the ReLU of the activation maps weighted by the
mean gradient of the target class w.r.t. those activations.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        # full backward hook is the modern, non-deprecated API
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, input_tensor, class_idx: int = None):
        """Return (cam[H,W] in [0,1], predicted_class_idx, probabilities)."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        # weights = global-average-pooled gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # [B,C,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.cpu().numpy(), class_idx, probs.detach().cpu().numpy()[0]


def overlay_heatmap(pil_image, cam, alpha: float = 0.45):
    """Blend a [0,1] CAM over a PIL image; returns a PIL RGB image."""
    import numpy as np
    from PIL import Image
    try:
        import cv2
        heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    except Exception:
        # fallback colormap without OpenCV: red-scaled heat
        h = (cam * 255).astype("uint8")
        heat = np.stack([h, np.zeros_like(h), 255 - h], axis=-1)

    base = pil_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    base_arr = np.asarray(base).astype("float32")
    blended = (1 - alpha) * base_arr + alpha * heat.astype("float32")
    return Image.fromarray(blended.clip(0, 255).astype("uint8"))
