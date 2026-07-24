"""Export the trained checkpoint to ONNX (+INT8) and run the latency benchmark.

Usage:
    python -m scripts.export
"""
from src.core.config import Config
from src.core.logging import logger
from src.models.model import build_model_from_checkpoint
from src.export.benchmark import benchmark


def main():
    model, _ = build_model_from_checkpoint(Config.CHECKPOINT_PATH, device="cpu")
    logger.info("Running export + benchmark")
    benchmark(model)


if __name__ == "__main__":
    main()
