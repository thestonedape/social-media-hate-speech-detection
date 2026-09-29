from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LABELS = ["hate", "offensive", "normal"]

# Drop-in locations checked at startup, in priority order:
#   1. model_store/transformer/  — a fine-tuned Hugging Face model directory
#   2. model_store/sklearn_pipeline.joblib — a TF-IDF + classifier pipeline
MODEL_STORE = Path(__file__).resolve().parent / "model_store"
TRANSFORMER_DIR = MODEL_STORE / "transformer"
SKLEARN_PIPELINE = MODEL_STORE / "sklearn_pipeline.joblib"

# Tiny lexicon so the placeholder gives varied, deterministic demo output.
# It is NOT a real classifier and is replaced by the trained model drop-in.
_OFFENSIVE_HINTS = re.compile(
    r"\b(stupid|idiot|dumb|trash|shut up|loser|ugly|pathetic)\b", re.IGNORECASE
)
_HATE_HINTS = re.compile(
    r"\b(go back to|subhuman|vermin|exterminate|not human|your kind)\b",
    re.IGNORECASE,
)


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_backend: str


class HateSpeechService:
    """Serves predictions from whichever model artifact is present.

    Placeholder mode keeps the API fully functional before a trained
    model is attached, so the frontend can be built and deployed now.
    """

    def __init__(self) -> None:
        self.backend = "placeholder"
        self._pipeline = None
        self._transformer = None
        self._tokenizer = None

        if TRANSFORMER_DIR.is_dir():
            self._load_transformer()
        elif SKLEARN_PIPELINE.exists():
            self._load_sklearn()

    def _load_transformer(self) -> None:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        self._transformer = AutoModelForSequenceClassification.from_pretrained(
            TRANSFORMER_DIR
        )
        self._transformer.eval()
        self.backend = "transformer"

    def _load_sklearn(self) -> None:
        import joblib

        self._pipeline = joblib.load(SKLEARN_PIPELINE)
        self.backend = "sklearn"

    def predict(self, text: str) -> PredictionResult:
        if self.backend == "transformer":
            probs = self._predict_transformer(text)
        elif self.backend == "sklearn":
            probs = self._predict_sklearn(text)
        else:
            probs = self._predict_placeholder(text)

        label = max(probs, key=probs.get)
        return PredictionResult(
            label=label,
            confidence=probs[label],
            probabilities=probs,
            model_backend=self.backend,
        )

    def _predict_transformer(self, text: str) -> dict[str, float]:
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        with torch.inference_mode():
            logits = self._transformer(**inputs).logits
        scores = torch.softmax(logits, dim=1).squeeze(0).tolist()
        id2label = getattr(self._transformer.config, "id2label", {}) or {}
        configured_labels = [
            id2label.get(index, id2label.get(str(index), f"LABEL_{index}"))
            for index in range(len(scores))
        ]
        labels = self._resolve_output_labels(configured_labels)
        return self._validated_probabilities(scores, labels)

    def _predict_sklearn(self, text: str) -> dict[str, float]:
        scores = self._pipeline.predict_proba([text])[0]
        configured_labels = getattr(self._pipeline, "classes_", None)
        labels = self._resolve_output_labels(configured_labels)
        return self._validated_probabilities(scores, labels)

    @staticmethod
    def _resolve_output_labels(values: Sequence[object] | None) -> list[str]:
        if values is None:
            return LABELS.copy()

        normalized = [str(value).strip().lower() for value in values]
        if normalized == ["0", "1", "2"]:
            return LABELS.copy()
        if all(label.startswith("label_") for label in normalized):
            return LABELS.copy()
        if len(normalized) == len(LABELS) and set(normalized) == set(LABELS):
            return normalized

        raise RuntimeError(
            "Model labels must map to hate, offensive, and normal."
        )

    @staticmethod
    def _validated_probabilities(
        scores: Sequence[float], labels: Sequence[str]
    ) -> dict[str, float]:
        if len(scores) != len(LABELS) or len(labels) != len(LABELS):
            raise RuntimeError("Model must return exactly three class scores.")

        numeric_scores = [float(score) for score in scores]
        if any(not math.isfinite(score) or score < 0 for score in numeric_scores):
            raise RuntimeError("Model returned invalid class probabilities.")

        total = sum(numeric_scores)
        if total <= 0:
            raise RuntimeError("Model returned an empty probability distribution.")

        probabilities = {
            label: score / total
            for label, score in zip(labels, numeric_scores, strict=True)
        }
        if set(probabilities) != set(LABELS):
            raise RuntimeError("Model output labels do not match the API contract.")
        return {label: probabilities[label] for label in LABELS}

    def _predict_placeholder(self, text: str) -> dict[str, float]:
        hate_hits = len(_HATE_HINTS.findall(text))
        offensive_hits = len(_OFFENSIVE_HINTS.findall(text))

        hate = 0.05 + 0.4 * min(hate_hits, 2)
        offensive = 0.10 + 0.3 * min(offensive_hits, 2)
        normal = max(0.05, 1.0 - hate - offensive)

        total = hate + offensive + normal
        return {
            "hate": round(hate / total, 4),
            "offensive": round(offensive / total, 4),
            "normal": round(normal / total, 4),
        }
