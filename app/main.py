from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.feedback import (
    FeedbackError,
    InputGuardrailError,
    MissingAPIKeyError,
    UpstreamAuthError,
    UpstreamBadResponseError,
    UpstreamRateLimitError,
    UpstreamTemporaryError,
    UpstreamTimeoutError,
    get_feedback,
)
from app.models import FeedbackRequest, FeedbackResponse

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response.headers["x-request-id"] = request_id
    return response


def _validate_runtime_configuration() -> None:
    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))
    max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "700"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    cache_max_entries = int(os.getenv("CACHE_MAX_ENTRIES", "256"))
    if timeout <= 0:
        raise RuntimeError("OPENAI_TIMEOUT_SECONDS must be greater than 0")
    if max_output_tokens < 64:
        raise RuntimeError("OPENAI_MAX_OUTPUT_TOKENS must be at least 64")
    if max_retries < 0:
        raise RuntimeError("OPENAI_MAX_RETRIES must be greater than or equal to 0")
    if cache_max_entries <= 0:
        raise RuntimeError("CACHE_MAX_ENTRIES must be greater than 0")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _validate_runtime_configuration()
    yield


app = FastAPI(
    title="Language Feedback API",
    description="Analyzes learner-written sentences and provides structured language feedback.",
    version="1.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "%s %s request_id=%s duration_ms=%s",
            request.method,
            request.url.path,
            request_id,
            duration_ms,
        )
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        request,
        status_code=422,
        code="validation_error",
        message=str(exc),
    )


@app.exception_handler(FeedbackError)
async def feedback_exception_handler(request: Request, exc: FeedbackError) -> JSONResponse:
    status_code = 500
    error_code = "internal_error"

    if isinstance(exc, InputGuardrailError):
        status_code = 400
        error_code = "invalid_input"
    elif isinstance(exc, MissingAPIKeyError):
        status_code = 500
        error_code = "missing_api_key"
    elif isinstance(exc, UpstreamAuthError):
        status_code = 502
        error_code = "upstream_auth_error"
    elif isinstance(exc, UpstreamRateLimitError):
        status_code = 429
        error_code = "upstream_rate_limited"
    elif isinstance(exc, UpstreamTimeoutError):
        status_code = 504
        error_code = "upstream_timeout"
    elif isinstance(exc, UpstreamTemporaryError):
        status_code = 503
        error_code = "upstream_unavailable"
    elif isinstance(exc, UpstreamBadResponseError):
        status_code = 502
        error_code = "upstream_bad_response"

    logger.warning(
        "Feedback error request_id=%s code=%s message=%s",
        getattr(request.state, "request_id", None),
        error_code,
        str(exc),
    )
    return _error_response(request, status_code, error_code, str(exc))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    return await get_feedback(request)
