"""Train MedVision.

Usage:
    python -m scripts.train                 # train with defaults (.env / env vars)
    EPOCHS=20 BACKBONE=vit_small_patch16_224 python -m scripts.train
"""
import json

from src.core.config import Config
from src.core.logging import logger
from src.data.dataset import get_datasets, get_dataloaders, compute_class_weights
from src.models.model import build_model
from src.training.trainer import Trainer


def main():
    train_ds, _, _ = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders()
    class_weights = compute_class_weights(train_ds) if Config.USE_CLASS_WEIGHTS else None

    model = build_model()
    trainer = Trainer(model, train_loader, val_loader, class_weights=class_weights)
    trainer.fit()

    logger.info("Evaluating best model on the test split")
    test_metrics, cm = trainer.evaluate(test_loader)
    logger.info(f"TEST metrics: {test_metrics}")

    report = {"config": Config.summary(), "test_metrics": test_metrics,
              "confusion_matrix": cm, "labels": Config.labels()}
    with open("artifacts/test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved artifacts/test_report.json")


if __name__ == "__main__":
    main()
