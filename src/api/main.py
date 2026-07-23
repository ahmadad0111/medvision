import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.logging import logger
from src.api.routes import health, predict


def create_app() -> FastAPI:
    app = FastAPI(
        title="MedVision API",
        version="1.0.0",
        description="Explainable skin-lesion classification (DermaMNIST) with Grad-CAM.",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(predict.router)

    frontend = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
    if os.path.isdir(frontend):
        from fastapi.staticfiles import StaticFiles
        app.mount("/app", StaticFiles(directory=frontend, html=True), name="frontend")

    @app.on_event("startup")
    def _startup():
        logger.info("MedVision API starting up")

    return app


app = create_app()
