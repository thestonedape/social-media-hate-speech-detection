from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    from .model import LABELS, HateSpeechService
except ImportError:
    from model import LABELS, HateSpeechService

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)
FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", DEFAULT_FRONTEND_ORIGINS).split(",")
    if origin.strip()
]

MAX_TEXT_CHARS = 2000
logger = logging.getLogger(__name__)

app = FastAPI(title="Hate Speech Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

service = HateSpeechService()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be empty.")
        return value


class HealthResponse(BaseModel):
    status: str
    model_backend: str
    labels: list[str]


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_backend: str


@app.get("/")
def root():
    return {"name": "Hate Speech Detection API", "status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_backend=service.backend,
        labels=LABELS,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    try:
        result = service.predict(request.text)
    except Exception as exc:
        logger.exception("Prediction failed using backend %s.", service.backend)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    return PredictionResponse(
        text=request.text,
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_backend=result.model_backend,
    )
