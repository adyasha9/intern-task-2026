from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  
    AsyncOpenAI = None

from app.models import (
    FeedbackRequest,
    FeedbackResponse,
    VALID_DIFFICULTIES,
    VALID_ERROR_TYPES,
)

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
"""

REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "700"))


_CACHE: dict[str, FeedbackResponse] = {}


def _cache_key(request: FeedbackRequest) -> str:
    payload = json.dumps(request.model_dump(mode="python"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Fallback for SDK variants.
    output = getattr(response, "output", None) or []
    collected: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                collected.append(text)
    if collected:
        return "\n".join(collected)

    raise ValueError("Model response did not contain text output")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Model did not return valid JSON")
        return json.loads(stripped[start : end + 1])


def _sanitize_error(error: dict[str, Any]) -> dict[str, str]:
    original = str(error.get("original", "")).strip()
    correction = str(error.get("correction", "")).strip()
    explanation = str(error.get("explanation", "")).strip()
    error_type = str(error.get("error_type", "other")).strip()

    if error_type not in VALID_ERROR_TYPES:
        error_type = "other"

    if not original:
        original = correction or ""
    if not correction:
        correction = original
    if not explanation:
        explanation = "Please compare the original and corrected forms."

    return {
        "original": original,
        "correction": correction,
        "error_type": error_type,
        "explanation": explanation,
    }


def _sanitize_response(data: dict[str, Any], request: FeedbackRequest) -> FeedbackResponse:
    corrected_sentence = str(data.get("corrected_sentence", request.sentence)).strip()
    difficulty = str(data.get("difficulty", "A1")).strip()
    errors_raw = data.get("errors", [])
    is_correct = bool(data.get("is_correct", False))

    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "A1"

    if not isinstance(errors_raw, list):
        errors_raw = []

    errors = [_sanitize_error(item) for item in errors_raw if isinstance(item, dict)]

    if is_correct:
        corrected_sentence = request.sentence
        errors = []
    elif not errors and corrected_sentence == request.sentence:
        is_correct = True
    elif not errors and corrected_sentence != request.sentence:
        errors = [
            {
                "original": request.sentence,
                "correction": corrected_sentence,
                "error_type": "other",
                "explanation": f"The sentence was adjusted to sound natural in {request.target_language}.",
            }
        ]

    return FeedbackResponse(
        corrected_sentence=corrected_sentence,
        is_correct=is_correct,
        errors=errors,
        difficulty=difficulty,
    )


async def _call_openai(request: FeedbackRequest) -> dict[str, Any]:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is not installed. Install dependencies from requirements.txt.")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    key = _cache_key(request)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add a valid key to .env or the environment."
        )

    try:
        data = await asyncio.wait_for(_call_openai(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("LLM request timed out") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    result = _sanitize_response(data, request)
    _CACHE[key] = result
    return result
