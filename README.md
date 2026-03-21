# Language Feedback API

LLM-powered API for learner sentence correction and structured feedback. Built for the Pangea Chat Gen AI Intern Task (Summer 2026).

## Overview

This project exposes a FastAPI service with two endpoints:

- `GET /health` — health check
- `POST /feedback` — analyzes a learner-written sentence and returns:
  - a minimally corrected sentence
  - whether the sentence is correct
  - a list of structured errors
  - a CEFR difficulty estimate

The implementation is designed to be simple, schema-safe, and production-conscious.

## Design Decisions

### Stack
- **Python**
- **FastAPI**
- **OpenAI API**
- **Pydantic** for request/response validation
- **Pytest** for testing

### Why this design
I chose FastAPI because it provides clean request validation, straightforward endpoint definitions, and easy JSON response handling. Since the task is centered on structured output, Pydantic models and explicit response validation help reduce schema failures.

I used an LLM-based approach instead of rule-based grammar logic because the task needs to support multiple languages, including non-Latin scripts. A single well-designed multilingual prompt is more scalable than language-specific heuristics.

## Prompt Strategy

The prompt is designed to make the model behave like a language-learning feedback engine rather than a general chatbot.

Key prompt constraints:
- return **only valid JSON**
- make **minimal edits**
- preserve the learner’s voice
- keep explanations learner-friendly
- write explanations in the **native language**
- use only the allowed `error_type` values
- use only the allowed CEFR values (`A1`–`C2`)
- if the input is already correct:
  - keep the sentence unchanged
  - return `is_correct: true`
  - return an empty `errors` array

Because LLMs can sometimes produce extra text, the implementation includes post-processing that extracts JSON safely and validates it against the response model before returning it.

## Reliability / Production Feasibility

This submission includes a few practical safeguards:

- **Schema-safe parsing**: attempts to extract valid JSON even if the model adds wrapper text
- **Response validation**: final output is validated through Pydantic
- **Timeout-aware error handling**: failures are surfaced cleanly instead of silently returning malformed data
- **Simple in-memory caching**: repeated identical requests can be reused to reduce latency and token cost

If this were extended for production, I would add:
- Redis or persistent caching
- retry logic with exponential backoff for transient API failures
- request/response logging with redaction
- fallback model routing
- offline evaluation set for multilingual regression testing

## Assumptions

- The evaluator will provide a valid API key through `.env`
- The hidden evaluation emphasizes schema consistency and multilingual correctness
- Minimal corrections are preferred over aggressive rewrites

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── feedback.py
│   ├── main.py
│   └── models.py
├── tests/
│   ├── __init__.py
│   ├── test_feedback_integration.py
│   ├── test_feedback_unit.py
│   └── test_schema.py
├── schema/
│   ├── request.schema.json
│   └── response.schema.json
├── examples/
│   └── sample_inputs.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
