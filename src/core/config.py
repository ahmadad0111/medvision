"""Environment-driven configuration for MedVision.

Kept import-light (only stdlib) so it can be imported in tests without torch.
"""
import os


def _as_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


# DermaMNIST (HAM10000-derived) label map
DERMAMNIST_LABELS = {
    0: "actinic keratoses / intraepithelial carcinoma",
    1: "basal cell carcinoma",
    2: "benign keratosis-like lesions",
    3: "dermatofibroma",
    4: "melanoma",
    5: "melanocytic nevi",
    6: "vascular lesions",
}


class Config:
    # --- data ---
    DATASET = os.getenv("DATASET", "dermamnist")     # a MedMNIST 2D dataset flag
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "7"))
    IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
    DATA_ROOT = os.getenv("DATA_ROOT", "data")

    # ImageNet normalisation (pretrained backbones expect this)
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # --- model ---
    BACKBONE = os.getenv("BACKBONE", "convnext_tiny")  # any timm model
    PRETRAINED = _as_bool("PRETRAINED", True)
    DROPOUT = float(os.getenv("DROPOUT", "0.1"))

    # --- training ---
    EPOCHS = int(os.getenv("EPOCHS", "15"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
    LR = float(os.getenv("LR", "3e-4"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    USE_CLASS_WEIGHTS = _as_bool("USE_CLASS_WEIGHTS", True)
    WEIGHT_POWER = float(os.getenv("WEIGHT_POWER", "0.5"))  # 0=uniform, 0.5=sqrt (gentle), 1=full inverse
    LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.1"))
    SEED = int(os.getenv("SEED", "42"))

    # --- artifacts ---
    ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/model.pt")
    ONNX_PATH = os.getenv("ONNX_PATH", "artifacts/model.onnx")
    ONNX_INT8_PATH = os.getenv("ONNX_INT8_PATH", "artifacts/model.int8.onnx")

    # --- experiment tracking ---
    USE_WANDB = _as_bool("USE_WANDB", False)
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "medvision")

    # --- runtime ---
    DEVICE = os.getenv("DEVICE", "auto")  # auto|cpu|cuda

    @classmethod
    def labels(cls):
        return DERMAMNIST_LABELS

    @classmethod
    def summary(cls) -> dict:
        return {
            "dataset": cls.DATASET,
            "backbone": cls.BACKBONE,
            "num_classes": cls.NUM_CLASSES,
            "img_size": cls.IMG_SIZE,
            "device": cls.DEVICE,
        }


config = Config()
