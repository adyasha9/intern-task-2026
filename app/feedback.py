from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
except ImportError:  # pragma: no cover - handled at runtime in environments without deps
    APIConnectionError = None
    APIStatusError = None
    APITimeoutError = None
    AsyncOpenAI = None
    RateLimitError = None

from app.models import (
    FeedbackRequest,
    FeedbackResponse,
    VALID_DIFFICULTIES,
    VALID_ERROR_TYPES,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert multilingual language-learning feedback engine.

Your task is to analyze one learner-written sentence and return ONLY valid JSON.

Output contract:
{
  "corrected_sentence": "string",
  "is_correct": true,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "grammar",
      "explanation": "string"
    }
  ],
  "difficulty": "A1"
}

Hard rules:
1. Return JSON only. No markdown. No commentary.
2. Make minimal edits that preserve the learner's voice and meaning.
3. If the sentence is already correct, return:
   - corrected_sentence equal to the original sentence exactly
   - is_correct = true
   - errors = []
4. Explanations must be in the learner's native language.
5. Allowed error_type values only:
   grammar, spelling, word_choice, punctuation, word_order,
   missing_word, extra_word, conjugation, gender_agreement,
   number_agreement, tone_register, other
6. Allowed difficulty values only: A1, A2, B1, B2, C1, C2
7. Difficulty is based on sentence complexity, not number of mistakes.
8. Each error should align to a real edit in the corrected sentence.
9. Keep explanations concise, learner-friendly, and specific.
10. Support any language and any writing system.
11. Never invent an error if the sentence is already correct.
12. If there are edits, provide at least one error entry describing them.
"""

REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "700"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "256"))
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
INITIAL_RETRY_DELAY_SECONDS = float(os.getenv("OPENAI_INITIAL_RETRY_DELAY_SECONDS", "0.5"))


class FeedbackError(Exception):
    """Base error for feedback generation."""


class MissingAPIKeyError(FeedbackError):
    pass


class UpstreamTimeoutError(FeedbackError):
    pass


class UpstreamAuthError(FeedbackError):
    pass


class UpstreamRateLimitError(FeedbackError):
    pass


class UpstreamTemporaryError(FeedbackError):
    pass


class UpstreamBadResponseError(FeedbackError):
    pass


class InputGuardrailError(FeedbackError):
    pass


@dataclass(slots=True)
class CacheEntry:
    value: FeedbackResponse
    expires_at: float


_CACHE: "OrderedDict[str, CacheEntry]" = OrderedDict()
_CLIENT: AsyncOpenAI | None = None


def _cache_key(request: FeedbackRequest) -> str:
    payload = json.dumps(request.model_dump(mode="python"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_cached(key: str) -> FeedbackResponse | None:
    entry = _CACHE.get(key)
    now = time.time()
    if entry is None:
        return None
    if entry.expires_at <= now:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return entry.value


def _set_cached(key: str, value: FeedbackResponse) -> None:
    _CACHE[key] = CacheEntry(value=value, expires_at=time.time() + CACHE_TTL_SECONDS)
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_MAX_ENTRIES:
        _CACHE.popitem(last=False)


def _get_client() -> AsyncOpenAI:
    global _CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY is not set. Add a valid key to .env or the environment.")
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is not installed. Install dependencies from requirements.txt.")
    if _CLIENT is None:
        _CLIENT = AsyncOpenAI(api_key=api_key)
    return _CLIENT



def _extract_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    output = getattr(response, "output", None) or []
    collected: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                collected.append(text)
    if collected:
        return "\n".join(collected)

    raise UpstreamBadResponseError("Model response did not contain text output")



def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        for start_index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                parsed, end_index = decoder.raw_decode(stripped[start_index:])
            except json.JSONDecodeError:
                continue
            trailing = stripped[start_index + end_index :].strip()
            if isinstance(parsed, dict) and not trailing:
                return parsed
        raise UpstreamBadResponseError("Model did not return valid JSON") from exc

    if not isinstance(parsed, dict):
        raise UpstreamBadResponseError("Model JSON response must be an object")
    return parsed



def _sanitize_error(error: dict[str, Any], request: FeedbackRequest) -> dict[str, str]:
    original = str(error.get("original", "")).strip()
    correction = str(error.get("correction", "")).strip()
    explanation = str(error.get("explanation", "")).strip()
    error_type = str(error.get("error_type", "other")).strip()

    if error_type not in VALID_ERROR_TYPES:
        error_type = "other"

    if not original and not correction:
        raise UpstreamBadResponseError("Each error must include original or correction text")

    if not original:
        original = correction
    if not correction:
        correction = original
    if not explanation:
        explanation = f"Please explain this correction in {request.native_language}."

    return {
        "original": original,
        "correction": correction,
        "error_type": error_type,
        "explanation": explanation,
    }



def _response_has_meaningful_change(original: str, corrected: str) -> bool:
    return original.strip() != corrected.strip()



def _sanitize_response(data: dict[str, Any], request: FeedbackRequest) -> FeedbackResponse:
    corrected_sentence = str(data.get("corrected_sentence", request.sentence)).strip() or request.sentence
    difficulty = str(data.get("difficulty", "A1")).strip()
    errors_raw = data.get("errors", [])
    is_correct = bool(data.get("is_correct", False))

    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "A1"

    if not isinstance(errors_raw, list):
        raise UpstreamBadResponseError("Model response field 'errors' must be a list")

    errors = [_sanitize_error(item, request) for item in errors_raw if isinstance(item, dict)]
    sentence_changed = _response_has_meaningful_change(request.sentence, corrected_sentence)

    if is_correct:
        corrected_sentence = request.sentence
        errors = []
    elif not sentence_changed:
        if errors:
            corrected_sentence = request.sentence
        else:
            is_correct = True
    elif sentence_changed and not errors:
        errors = [
            {
                "original": request.sentence,
                "correction": corrected_sentence,
                "error_type": "other",
                "explanation": f"Please explain the correction in {request.native_language}.",
            }
        ]

    return FeedbackResponse(
        corrected_sentence=corrected_sentence,
        is_correct=is_correct,
        errors=errors,
        difficulty=difficulty,
    )



def _validate_input_guardrails(request: FeedbackRequest) -> None:
    normalized_sentence = request.sentence.strip()
    if not normalized_sentence:
        raise InputGuardrailError("Sentence must not be blank")
    if len(normalized_sentence) > 2000:
        raise InputGuardrailError("Sentence exceeds maximum supported length")
    lowered = normalized_sentence.lower()
    suspicious_markers = [
        "ignore previous instructions",
        "system prompt",
        "developer message",
        "return yaml",
        "```",
    ]
    if any(marker in lowered for marker in suspicious_markers):
        raise InputGuardrailError("Sentence appears to contain prompt-injection content")


async def _call_openai_once(request: FeedbackRequest) -> dict[str, Any]:
    client = _get_client()
    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}"
    )

    response = await client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
    )

    text = _extract_text(response)
    return _extract_json(text)


async def _call_openai_with_retries(request: FeedbackRequest) -> dict[str, Any]:
    delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await _call_openai_once(request)
        except MissingAPIKeyError:
            raise
        except UpstreamBadResponseError:
            raise
        except Exception as exc:  # noqa: BLE001 - explicit translation below
            is_last_attempt = attempt >= MAX_RETRIES
            translated = _translate_openai_error(exc)
            if isinstance(translated, (UpstreamBadResponseError, UpstreamAuthError)) or is_last_attempt:
                raise translated from exc
            logger.warning("Transient upstream error on attempt %s/%s: %s", attempt + 1, MAX_RETRIES + 1, translated)
            await asyncio.sleep(delay)
            delay *= 2

    raise UpstreamTemporaryError("OpenAI request failed")



def _translate_openai_error(exc: Exception) -> FeedbackError:
    if isinstance(exc, FeedbackError):
        return exc
    if isinstance(exc, TimeoutError):
        return UpstreamTimeoutError("LLM request timed out")
    if APITimeoutError is not None and isinstance(exc, APITimeoutError):
        return UpstreamTimeoutError("OpenAI request timed out")
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return UpstreamRateLimitError("OpenAI rate limit exceeded")
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        if exc.status_code in {401, 403}:
            return UpstreamAuthError("OpenAI authentication failed")
        if exc.status_code == 429:
            return UpstreamRateLimitError("OpenAI rate limit exceeded")
        if 500 <= exc.status_code < 600:
            return UpstreamTemporaryError(f"OpenAI service error ({exc.status_code})")
        return UpstreamBadResponseError(f"OpenAI returned an unexpected status ({exc.status_code})")
    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return UpstreamTemporaryError("OpenAI connection failed")
    return UpstreamTemporaryError(f"OpenAI request failed: {exc}")


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    _validate_input_guardrails(request)

    key = _cache_key(request)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        data = await asyncio.wait_for(_call_openai_with_retries(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError("LLM request timed out") from exc

    result = _sanitize_response(data, request)
    _set_cached(key, result)
    return result
