"""Load the Predictor once (lazily) and share it across requests."""
from functools import lru_cache

from src.core.config import Config
from src.core.logging import logger


@lru_cache(maxsize=1)
def get_predictor():
    from src.inference.predictor import Predictor
    logger.info(f"Loading model for serving: {Config.summary()}")
    return Predictor()
