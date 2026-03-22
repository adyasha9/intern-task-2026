from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AsyncOpenAI,
        RateLimitError,
    )
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

MAX_SENTENCE_LENGTH = 2000
MAX_LANGUAGE_LENGTH = 80
_DEFAULT_SUSPICIOUS_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "assistant message",
    "return yaml",
    "return markdown",
    "```",
)


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


_CACHE: OrderedDict[str, CacheEntry] = OrderedDict()
_IN_FLIGHT: dict[str, asyncio.Task[FeedbackResponse]] = {}
_CLIENT: AsyncOpenAI | None = None


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid float env var %s=%r; falling back to %s", name, raw, default)
        return default
    return value if value > 0 else default


def _get_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid int env var %s=%r; falling back to %s", name, raw, default)
        return default
    return value if value >= minimum else default


def _request_timeout_seconds() -> float:
    return _get_float_env("OPENAI_TIMEOUT_SECONDS", 25.0)


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"


def _max_output_tokens() -> int:
    return _get_int_env("OPENAI_MAX_OUTPUT_TOKENS", 700, minimum=64)


def _cache_ttl_seconds() -> int:
    return _get_int_env("CACHE_TTL_SECONDS", 300, minimum=1)


def _cache_max_entries() -> int:
    return _get_int_env("CACHE_MAX_ENTRIES", 256, minimum=1)


def _max_retries() -> int:
    return _get_int_env("OPENAI_MAX_RETRIES", 2, minimum=0)


def _initial_retry_delay_seconds() -> float:
    return _get_float_env("OPENAI_INITIAL_RETRY_DELAY_SECONDS", 0.5)


def _cache_key(request: FeedbackRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="python"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prune_cache(now: float | None = None) -> None:
    if now is None:
        now = time.time()
    expired = [key for key, entry in _CACHE.items() if entry.expires_at <= now]
    for key in expired:
        _CACHE.pop(key, None)
    max_entries = _cache_max_entries()
    while len(_CACHE) > max_entries:
        _CACHE.popitem(last=False)


def _get_cached(key: str) -> FeedbackResponse | None:
    now = time.time()
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if entry.expires_at <= now:
        _CACHE.pop(key, None)
        logger.debug("Cache expired for key=%s", key)
        return None
    _CACHE.move_to_end(key)
    logger.debug("Cache hit for key=%s", key)
    return entry.value


def _set_cached(key: str, value: FeedbackResponse) -> None:
    _prune_cache()
    _CACHE[key] = CacheEntry(value=value, expires_at=time.time() + _cache_ttl_seconds())
    _CACHE.move_to_end(key)
    _prune_cache()


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


def _normalize_for_compare(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)

    output = getattr(response, "output", None) or []
    collected: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                collected.append(str(text))
    if collected:
        return "\n".join(collected)

    raise UpstreamBadResponseError("Model response did not contain text output")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise UpstreamBadResponseError("Model returned an empty response")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise UpstreamBadResponseError("Model JSON response must be an object")
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in _JSON_OBJECT_RE.finditer(stripped):
        candidate = match.group(0)
        try:
            parsed, end_index = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if end_index == len(candidate) and isinstance(parsed, dict):
            return parsed

    raise UpstreamBadResponseError("Model did not return valid JSON")


def _error_aligns_with_edit(error: dict[str, str], original_sentence: str, corrected_sentence: str) -> bool:
    original = _normalize_for_compare(error["original"])
    correction = _normalize_for_compare(error["correction"])
    normalized_original_sentence = _normalize_for_compare(original_sentence)
    normalized_corrected_sentence = _normalize_for_compare(corrected_sentence)
    if original and original in normalized_original_sentence:
        return True
    if correction and correction in normalized_corrected_sentence:
        return True
    return False


def _sanitize_error(
    error: dict[str, Any],
    request: FeedbackRequest,
    *,
    corrected_sentence: str,
) -> dict[str, str]:
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

    if len(original) > 500 or len(correction) > 500:
        raise UpstreamBadResponseError("Error spans are too long")

    if not explanation:
        explanation = f"Review the correction in {request.native_language}."

    cleaned = {
        "original": original,
        "correction": correction,
        "error_type": error_type,
        "explanation": explanation,
    }
    if not _error_aligns_with_edit(cleaned, request.sentence, corrected_sentence):
        raise UpstreamBadResponseError("Model returned an error entry that does not align to the sentence edit")
    return cleaned


def _response_has_meaningful_change(original: str, corrected: str) -> bool:
    return _normalize_for_compare(original) != _normalize_for_compare(corrected)


def _deduplicate_errors(errors: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for error in errors:
        key = (error["original"], error["correction"], error["error_type"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(error)
    return deduped


def _sanitize_response(data: dict[str, Any], request: FeedbackRequest) -> FeedbackResponse:
    corrected_sentence = str(data.get("corrected_sentence", request.sentence)).strip() or request.sentence
    difficulty = str(data.get("difficulty", "A1")).strip()
    errors_raw = data.get("errors", [])
    is_correct = bool(data.get("is_correct", False))

    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "A1"

    if not isinstance(errors_raw, list):
        raise UpstreamBadResponseError("Model response field 'errors' must be a list")

    sentence_changed = _response_has_meaningful_change(request.sentence, corrected_sentence)
    if is_correct:
        corrected_sentence = request.sentence
        errors: list[dict[str, str]] = []
    else:
        errors = [
            _sanitize_error(item, request, corrected_sentence=corrected_sentence)
            for item in errors_raw
            if isinstance(item, dict)
        ]
        errors = _deduplicate_errors(errors)

        if sentence_changed and not errors:
            errors = [
                {
                    "original": request.sentence,
                    "correction": corrected_sentence,
                    "error_type": "other",
                    "explanation": f"Review the correction in {request.native_language}.",
                }
            ]
        elif not sentence_changed and errors:
            corrected_sentence = request.sentence
        elif not sentence_changed and not errors:
            is_correct = True

    response = FeedbackResponse(
        corrected_sentence=corrected_sentence,
        is_correct=is_correct,
        errors=errors,
        difficulty=difficulty,
    )

    if response.is_correct and response.corrected_sentence != request.sentence:
        raise UpstreamBadResponseError("Correct sentences must preserve the original sentence exactly")
    if response.is_correct and response.errors:
        raise UpstreamBadResponseError("Correct sentences must not contain errors")
    if not response.is_correct and _response_has_meaningful_change(request.sentence, response.corrected_sentence) and not response.errors:
        raise UpstreamBadResponseError("Changed sentences must include at least one error entry")

    return response


def _validate_input_guardrails(request: FeedbackRequest) -> None:
    normalized_sentence = request.sentence.strip()
    if not normalized_sentence:
        raise InputGuardrailError("Sentence must not be blank")
    if len(normalized_sentence) > MAX_SENTENCE_LENGTH:
        raise InputGuardrailError("Sentence exceeds maximum supported length")
    if len(request.target_language) > MAX_LANGUAGE_LENGTH or len(request.native_language) > MAX_LANGUAGE_LENGTH:
        raise InputGuardrailError("Language values exceed maximum supported length")

    lowered = normalized_sentence.lower()
    if any(marker in lowered for marker in _DEFAULT_SUSPICIOUS_MARKERS):
        raise InputGuardrailError("Sentence appears to contain prompt-injection content")

    newline_count = normalized_sentence.count("\n")
    if newline_count > 8:
        raise InputGuardrailError("Sentence contains too many line breaks")

    if normalized_sentence.count("http://") + normalized_sentence.count("https://") > 3:
        raise InputGuardrailError("Sentence contains too many URLs")

    if re.search(r"(?:[A-Za-z0-9+/]{80,}={0,2})", normalized_sentence):
        raise InputGuardrailError("Sentence appears to contain encoded or non-linguistic content")

    symbol_ratio = sum(1 for char in normalized_sentence if not char.isalnum() and not char.isspace()) / max(len(normalized_sentence), 1)
    if symbol_ratio > 0.45 and len(normalized_sentence) > 40:
        raise InputGuardrailError("Sentence appears to contain mostly symbols rather than natural language")


async def _call_openai_once(request: FeedbackRequest) -> dict[str, Any]:
    client = _get_client()
    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}"
    )

    logger.debug(
        "Calling OpenAI model=%s sentence_length=%s target_language=%s native_language=%s",
        _model_name(),
        len(request.sentence),
        request.target_language,
        request.native_language,
    )
    try:
        response = await asyncio.wait_for(
            client.responses.create(
                model=_model_name(),
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_output_tokens=_max_output_tokens(),
                temperature=0,
            ),
            timeout=_request_timeout_seconds(),
        )
    except TimeoutError as exc:
        raise UpstreamTimeoutError("Upstream model request timed out") from exc
    except Exception as exc:
        if APITimeoutError is not None and isinstance(exc, APITimeoutError):
            raise UpstreamTimeoutError("Upstream model request timed out") from exc
        if RateLimitError is not None and isinstance(exc, RateLimitError):
            raise UpstreamRateLimitError("Upstream model rate limit exceeded") from exc
        if APIStatusError is not None and isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code in {401, 403}:
                raise UpstreamAuthError("Upstream model authentication failed") from exc
            if status_code == 429:
                raise UpstreamRateLimitError("Upstream model rate limit exceeded") from exc
            if status_code in {500, 502, 503, 504}:
                raise UpstreamTemporaryError("Upstream model is temporarily unavailable") from exc
            raise UpstreamBadResponseError(f"Upstream model request failed with status code {status_code}") from exc
        if APIConnectionError is not None and isinstance(exc, APIConnectionError):
            raise UpstreamTemporaryError("Unable to reach upstream model service") from exc
        raise UpstreamTemporaryError("Unexpected upstream model failure") from exc

    text = _extract_text(response)
    return _extract_json(text)


async def _call_openai_with_retries(request: FeedbackRequest) -> dict[str, Any]:
    delay = _initial_retry_delay_seconds()
    max_retries = _max_retries()
    for attempt in range(max_retries + 1):
        try:
            return await _call_openai_once(request)
        except (MissingAPIKeyError, UpstreamAuthError, UpstreamBadResponseError, InputGuardrailError):
            raise
        except (UpstreamRateLimitError, UpstreamTemporaryError, UpstreamTimeoutError) as exc:
            if attempt >= max_retries:
                raise
            logger.warning(
                "Retrying upstream call after failure=%s attempt=%s/%s",
                exc.__class__.__name__,
                attempt + 1,
                max_retries + 1,
            )
            await asyncio.sleep(delay)
            delay *= 2

    raise UpstreamTemporaryError("Exhausted retries without receiving a response")


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    _validate_input_guardrails(request)
    key = _cache_key(request)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    existing_task = _IN_FLIGHT.get(key)
    if existing_task is not None:
        logger.debug("Joining in-flight request for key=%s", key)
        return await existing_task

    async def _compute() -> FeedbackResponse:
        started = time.perf_counter()
        data = await _call_openai_with_retries(request)
        response = _sanitize_response(data, request)
        _set_cached(key, response)
        logger.info(
            "Generated feedback in %.3fs sentence_length=%s target_language=%s",
            time.perf_counter() - started,
            len(request.sentence),
            request.target_language,
        )
        return response

    task = asyncio.create_task(_compute())
    _IN_FLIGHT[key] = task
    try:
        return await task
    finally:
        _IN_FLIGHT.pop(key, None)
