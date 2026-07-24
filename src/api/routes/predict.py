import time

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query

from src.api.schemas import PredictResponse
from src.api.dependencies import get_predictor
from src.core.logging import logger

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(True, description="Return a Grad-CAM heatmap overlay"),
    predictor=Depends(get_predictor),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    from PIL import Image
    import io
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    start = time.time()
    try:
        result = predictor.predict(image, explain=explain)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="No trained model found. Train first: python -m scripts.train")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result
