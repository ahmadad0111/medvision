"""Slim MedVision edge API — ONNX inference only, no PyTorch.

Serves the same /predict contract as the full API (minus the Grad-CAM overlay),
so the existing web UI works against it. Intended for Jetson / edge devices.

    uvicorn src.api.edge_main:app --host 0.0.0.0 --port 8000
"""
import io
import os
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import Config
from src.core.logging import logger

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from src.inference.onnx_predictor import ONNXPredictor
        _predictor = ONNXPredictor()
    return _predictor


def create_edge_app() -> FastAPI:
    app = FastAPI(title="MedVision Edge (ONNX)", version="1.0.0",
                  description="Lightweight ONNX inference for Jetson / edge devices.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "healthy", "mode": "onnx-edge"}

    @app.get("/version")
    def version():
        return {"service": "medvision-edge", "onnx_path": Config.ONNX_PATH,
                "labels": Config.labels()}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload an image file.")
        from PIL import Image
        try:
            image = Image.open(io.BytesIO(await file.read()))
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read the image.")
        try:
            start = time.time()
            result = _get_predictor().predict(image)
            result["latency_ms"] = round((time.time() - start) * 1000, 1)
            return result
        except FileNotFoundError:
            raise HTTPException(status_code=503,
                                detail="No ONNX model found. Run `python -m scripts.export` first.")
        except Exception as exc:
            logger.exception("Edge prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    frontend = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
    if os.path.isdir(frontend):
        from fastapi.staticfiles import StaticFiles
        app.mount("/app", StaticFiles(directory=frontend, html=True), name="frontend")

    @app.on_event("startup")
    def _startup():
        logger.info("MedVision Edge API starting up (ONNX)")

    return app


app = create_edge_app()
