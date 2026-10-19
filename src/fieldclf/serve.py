"""HTTP inference service.

    FIELDCLF_MODEL_DIR=artifacts/tfidf_logreg uvicorn fieldclf.serve:app

The model is loaded once at start-up; requests are validated before they reach
it (batch size and text length are bounded so a single request cannot exhaust
memory).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from fieldclf import __version__
from fieldclf.predictor import Classifier

MAX_BATCH = 64
MAX_CHARS = 20_000


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)
    top_k: int = Field(default=5, ge=1, le=37)


class Score(BaseModel):
    field: str
    score: float
    threshold: float


class Prediction(BaseModel):
    fields: list[str]
    scores: list[Score]


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


def create_app(classifier: Classifier | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if classifier is not None:
            app.state.clf = classifier
        else:
            directory = Path(os.environ.get("FIELDCLF_MODEL_DIR", "artifacts/tfidf_logreg"))
            app.state.clf = Classifier.load(directory)
        # Pay one-off initialisation costs before the first real request
        # (cold first call: ~200 ms; warm: ~5 ms for the TF-IDF model).
        app.state.clf.predict(["warm-up"])
        yield

    app = FastAPI(title="CORDIS field classifier", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["Server-Timing"] = f"app;dur={1000 * (time.perf_counter() - t0):.1f}"
        return response

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        clf: Classifier = request.app.state.clf
        return {"status": "ok", "model": clf.model.name, "labels": len(clf.labels)}

    @app.get("/labels")
    def labels(request: Request) -> list[str]:
        return list(request.app.state.clf.labels)

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        clf: Classifier = request.app.state.clf
        for i, text in enumerate(body.texts):
            if not text.strip():
                raise HTTPException(422, f"texts[{i}] is empty")
            if len(text) > MAX_CHARS:
                raise HTTPException(413, f"texts[{i}] exceeds {MAX_CHARS} characters")
        out = clf.predict(body.texts, top_k=body.top_k)
        return PredictResponse(model=clf.model.name, predictions=out)  # type: ignore[arg-type]

    return app


app = create_app()
