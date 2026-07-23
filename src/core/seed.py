"""Reproducibility helper. Imports torch/numpy lazily so tests stay light."""


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(device: str = "auto"):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
