"""Training / evaluation loop for MedVision.

Handles class-imbalanced medical data (weighted loss + label smoothing),
tracks accuracy / macro-F1 / macro-AUROC each epoch, checkpoints the best
model by validation AUROC, and optionally logs to Weights & Biases.
"""
import os

from src.core.config import Config
from src.core.logging import logger
from src.core.seed import set_seed, resolve_device
from src.training.metrics import summarize, confusion_matrix


class Trainer:
    def __init__(self, model, train_loader, val_loader, class_weights=None):
        import torch
        set_seed(Config.SEED)
        self.device = resolve_device(Config.DEVICE)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        weight = None
        if Config.USE_CLASS_WEIGHTS and class_weights is not None:
            weight = class_weights.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=weight, label_smoothing=Config.LABEL_SMOOTHING)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS)
        self.best_auroc = -1.0
        self._wandb = self._init_wandb()

    def _init_wandb(self):
        if not Config.USE_WANDB:
            return None
        try:
            import wandb
            wandb.init(project=Config.WANDB_PROJECT, config=Config.summary())
            return wandb
        except Exception as exc:
            logger.warning(f"W&B disabled: {exc}")
            return None

    @staticmethod
    def _labels_to_1d(y):
        # MedMNIST targets are (B,1); flatten to (B,)
        return y.view(-1).long()

    def _run_epoch(self, loader, train: bool):
        import torch
        self.model.train(train)
        total_loss, preds, labels, probs = 0.0, [], [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for x, y in loader:
                x = x.to(self.device)
                y = self._labels_to_1d(y).to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                p = torch.softmax(logits, dim=1)
                preds += p.argmax(1).cpu().tolist()
                probs += p.detach().cpu().tolist()
                labels += y.cpu().tolist()
        metrics = summarize(preds, labels, probs, Config.NUM_CLASSES)
        metrics["loss"] = round(total_loss / max(1, len(labels)), 4)
        return metrics, (preds, labels)

    def fit(self):
        os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)
        for epoch in range(1, Config.EPOCHS + 1):
            train_m, _ = self._run_epoch(self.train_loader, train=True)
            val_m, (vp, vl) = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()
            logger.info(
                f"epoch {epoch:02d} | "
                f"train loss {train_m['loss']} acc {train_m['accuracy']} | "
                f"val acc {val_m['accuracy']} f1 {val_m['macro_f1']} "
                f"auroc {val_m['macro_auroc']}"
            )
            if self._wandb:
                self._wandb.log({"epoch": epoch,
                                 **{f"train/{k}": v for k, v in train_m.items()},
                                 **{f"val/{k}": v for k, v in val_m.items()}})
            if val_m["macro_auroc"] == val_m["macro_auroc"] and \
                    val_m["macro_auroc"] > self.best_auroc:
                self.best_auroc = val_m["macro_auroc"]
                self._save(epoch, val_m)
        logger.info(f"Best val AUROC: {self.best_auroc}")
        return self.best_auroc

    def _save(self, epoch, metrics):
        import torch
        torch.save({
            "model_state": self.model.state_dict(),
            "backbone": Config.BACKBONE,
            "num_classes": Config.NUM_CLASSES,
            "img_size": Config.IMG_SIZE,
            "epoch": epoch,
            "val_metrics": metrics,
        }, Config.CHECKPOINT_PATH)
        logger.info(f"Saved checkpoint -> {Config.CHECKPOINT_PATH} (AUROC {metrics['macro_auroc']})")

    def evaluate(self, loader):
        metrics, (preds, labels) = self._run_epoch(loader, train=False)
        cm = confusion_matrix(preds, labels, Config.NUM_CLASSES)
        return metrics, cm
