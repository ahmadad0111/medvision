"""timm backbone factory.

Any timm model works via BACKBONE (default convnext_tiny, which exposes clean
convolutional feature maps for Grad-CAM). The classifier head is resized to
NUM_CLASSES.
"""
from src.core.config import Config
from src.core.logging import logger


def build_model(backbone: str = None, num_classes: int = None,
                pretrained: bool = None):
    import timm
    backbone = backbone or Config.BACKBONE
    num_classes = num_classes or Config.NUM_CLASSES
    pretrained = Config.PRETRAINED if pretrained is None else pretrained
    try:
        model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes,
            drop_rate=Config.DROPOUT,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"timm could not create backbone '{backbone}': {exc}. "
            f"Your installed timm ({getattr(timm, '__version__', '?')}) may be too old for this model. "
            f"Fix: `pip install -U timm`, or set a backbone your version has, "
            f"e.g. BACKBONE=resnet50 (ImageNet-pretrained and Grad-CAM friendly)."
        ) from exc
    logger.info(f"Built {backbone} (pretrained={pretrained}, classes={num_classes})")
    return model


def find_gradcam_target_layer(model):
    """Return the last convolutional module — the standard Grad-CAM target.

    Works for ConvNeXt/ResNet/EfficientNet-style timm models. For pure ViTs
    (no conv layers) this returns None and the caller should pick a norm layer.
    """
    import torch.nn as nn
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def load_checkpoint(model, path: str, device: str = "cpu"):
    import torch
    state = torch.load(path, map_location=device)
    state = state.get("model_state", state)
    model.load_state_dict(state)
    model.eval()
    return model


def build_model_from_checkpoint(path: str, device: str = "cpu"):
    """Rebuild the exact architecture stored in the checkpoint, then load weights.

    The training checkpoint records `backbone`/`num_classes`, so serving and
    export never depend on the BACKBONE env var matching how you trained.
    """
    import torch
    ckpt = torch.load(path, map_location=device)
    backbone = ckpt.get("backbone", Config.BACKBONE)
    num_classes = ckpt.get("num_classes", Config.NUM_CLASSES)
    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint {path} (backbone={backbone}, classes={num_classes})")
    return model, ckpt
