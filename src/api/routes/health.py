from fastapi import APIRouter
from src.core.config import Config

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/version")
def version():
    return {"service": "medvision", "version": "1.0.0", "config": Config.summary(),
            "labels": Config.labels()}
