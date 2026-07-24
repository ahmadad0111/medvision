"""DermaMNIST data pipeline (via the MedMNIST package).

MedMNIST auto-downloads the dataset on first use, so this runs out of the box
with no manual data wrangling. Images are resized to the backbone's expected
input and normalised with ImageNet statistics.
"""
from src.core.config import Config
from src.core.logging import logger


def _build_transforms(train: bool):
    from torchvision import transforms
    aug = []
    if train:
        aug += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1),
        ]
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD),
    ])


def _dataset_class():
    import medmnist
    from medmnist import INFO
    info = INFO[Config.DATASET]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass, info


def get_datasets():
    import os
    DataClass, info = _dataset_class()
    logger.info(f"Loading {Config.DATASET}: {info['task']} ({len(info['label'])} classes)")
    os.makedirs(Config.DATA_ROOT, exist_ok=True)  # MedMNIST needs an existing root dir
    common = dict(root=Config.DATA_ROOT, download=True, as_rgb=True)
    train = DataClass(split="train", transform=_build_transforms(True), **common)
    val = DataClass(split="val", transform=_build_transforms(False), **common)
    test = DataClass(split="test", transform=_build_transforms(False), **common)
    return train, val, test


def get_dataloaders():
    from torch.utils.data import DataLoader
    train, val, test = get_datasets()
    dl = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=Config.BATCH_SIZE, shuffle=shuffle,
        num_workers=Config.NUM_WORKERS, pin_memory=True,
    )
    return dl(train, True), dl(val, False), dl(test, False)


def compute_class_weights(train_dataset):
    """Inverse-frequency class weights for imbalanced medical data."""
    import torch
    labels = [int(y) for _, y in _iter_labels(train_dataset)]
    counts = [0] * Config.NUM_CLASSES
    for y in labels:
        counts[y] += 1
    total = sum(counts)
    weights = [total / (Config.NUM_CLASSES * c) if c else 0.0 for c in counts]
    logger.info(f"Class counts: {counts}")
    return torch.tensor(weights, dtype=torch.float32)


def _iter_labels(dataset):
    # MedMNIST stores labels as shape (N,1); normalise to scalars
    for i in range(len(dataset)):
        _, y = dataset[i]
        try:
            y = int(y[0])
        except (TypeError, IndexError):
            y = int(y)
        yield None, y
