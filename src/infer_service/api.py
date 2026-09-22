"""FastAPI application: image classification with micro-batching, LLM gateway, probes and metrics."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from infer_service import __version__
from infer_service.batcher import MicroBatcher
from infer_service.config import Settings, get_settings
from infer_service.llm_gateway import LLMGateway
from infer_service.model import OnnxClassifier, load_labels
from infer_service.observability import (
    MODEL_LOADED,
    PREPROCESS_LATENCY,
    REQUEST_LATENCY,
    REQUESTS,
    configure_logging,
)
from infer_service.preprocess import image_to_tensor

log = logging.getLogger("infer_service.api")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    model: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.model = None
        app.state.batcher = None
        app.state.llm = LLMGateway(settings)
        if Path(settings.model_path).exists():
            model = OnnxClassifier(settings.model_path, load_labels(settings.labels_path),
                                   intra_op_threads=settings.intra_op_threads)
            model.warmup()
            batcher = MicroBatcher(model.predict, max_batch=settings.batch_max,
                                   max_wait_ms=settings.batch_wait_ms)
            await batcher.start()
            app.state.model, app.state.batcher = model, batcher
            MODEL_LOADED.set(1)
            log.info("model loaded path=%s input=%s providers=%s", model.model_path,
                     model.input_shape, model.providers)
        else:
            MODEL_LOADED.set(0)
            log.warning("model file not found path=%s; /v1/classify will return 503", settings.model_path)
        yield
        if app.state.batcher:
            await app.state.batcher.stop()

    app = FastAPI(title="Inference Service Bench", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = request.scope["route"].path if request.scope.get("route") else request.url.path
        REQUESTS.labels(route=route, status=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(route=route).observe(elapsed)
        response.headers["x-request-id"] = request_id
        log.info("request_id=%s method=%s route=%s status=%d ms=%.2f", request_id, request.method, route,
                 response.status_code, elapsed * 1000)
        return response

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request, response: Response) -> dict:
        model = request.app.state.model
        if model is None:
            response.status_code = 503
            return {"status": "model not loaded", "model_path": settings.model_path}
        return {"status": "ready", "model_path": model.model_path, "providers": model.providers}

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", tags=["vision"])
    async def model_info(request: Request) -> dict:
        model = request.app.state.model
        if model is None:
            raise HTTPException(503, "model not loaded")
        return {
            "path": model.model_path, "input_name": model.input_name, "input_shape": model.input_shape,
            "providers": model.providers, "labels": len(model.labels),
            "batching": {"max_batch": settings.batch_max, "max_wait_ms": settings.batch_wait_ms},
            "batches_run": request.app.state.batcher.batches_run,
        }

    @app.post("/v1/classify", tags=["vision"])
    async def classify(request: Request, file: UploadFile, top_k: int | None = None) -> dict:
        model, batcher = request.app.state.model, request.app.state.batcher
        if model is None:
            raise HTTPException(503, "model not loaded")
        data = await file.read()
        if not data:
            raise HTTPException(422, "empty file")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file larger than 10 MiB")
        t0 = time.perf_counter()
        try:
            tensor = image_to_tensor(data, settings.input_size)
        except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
            raise HTTPException(422, f"could not decode image: {exc.__class__.__name__}") from exc
        t1 = time.perf_counter()
        PREPROCESS_LATENCY.observe(t1 - t0)
        probs = await batcher.submit(tensor)
        t2 = time.perf_counter()
        return {
            "predictions": model.top_k(probs, top_k or settings.top_k),
            "timings_ms": {"preprocess": round((t1 - t0) * 1000, 2),
                           "queue_and_infer": round((t2 - t1) * 1000, 2)},
        }

    @app.post("/v1/llm/complete", tags=["llm"])
    async def llm_complete(body: CompletionRequest, request: Request) -> dict:
        gateway: LLMGateway = request.app.state.llm
        if not gateway.configured:
            raise HTTPException(503, "LLM gateway not configured (set OPENAI_API_KEY or INFER_LLM_API_KEY)")
        try:
            return await gateway.complete(body.prompt, max_tokens=body.max_tokens, model=body.model)
        except Exception as exc:  # noqa: BLE001 - upstream failures become 502s
            raise HTTPException(502, f"upstream error: {exc.__class__.__name__}") from exc

    return app


app = create_app()
