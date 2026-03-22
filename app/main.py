from __future__ import annotations

import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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

app = FastAPI(
    title="Language Feedback API",
    description="Analyzes learner-written sentences and provides structured language feedback.",
    version="1.1.0",
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


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

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    return await get_feedback(request)
