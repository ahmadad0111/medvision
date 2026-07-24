from typing import List, Optional
from pydantic import BaseModel


class Prediction(BaseModel):
    class_id: int
    label: str
    prob: float


class PredictResponse(BaseModel):
    prediction: Prediction
    top_k: List[Prediction]
    gradcam_png_base64: Optional[str] = None
    latency_ms: Optional[float] = None
