import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.feedback import (
    _CACHE,
    _CLIENT,
    _extract_json,
    get_feedback,
)
from app.main import app
from app.models import FeedbackRequest


def _mock_responses_api(response_data: dict | str) -> MagicMock:
    response = MagicMock()
    response.output_text = response_data if isinstance(response_data, str) else json.dumps(response_data)
    return response


@pytest.fixture(autouse=True)
def clear_cache_and_client() -> None:
    _CACHE.clear()
    import app.feedback as feedback_module

    feedback_module._CLIENT = None


@pytest.mark.asyncio
async def test_feedback_with_errors() -> None:
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        with patch("app.feedback.AsyncOpenAI") as mock_client:
            instance = mock_client.return_value
            instance.responses.create = AsyncMock(
                return_value=_mock_responses_api(mock_response)
            )

            request = FeedbackRequest(
                sentence="Yo soy fue al mercado ayer.",
                target_language="Spanish",
                native_language="English",
            )
            result = await get_feedback(request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence() -> None:
    mock_response = {
        "corrected_sentence": "Changed by model but should be overwritten",
        "is_correct": True,
        "errors": [
            {
                "original": "x",
                "correction": "y",
                "error_type": "grammar",
                "explanation": "Should be removed",
            }
        ],
        "difficulty": "B1",
    }

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        with patch("app.feedback.AsyncOpenAI") as mock_client:
            instance = mock_client.return_value
            instance.responses.create = AsyncMock(
                return_value=_mock_responses_api(mock_response)
            )

            request = FeedbackRequest(
                sentence="Ich habe gestern einen interessanten Film gesehen.",
                target_language="German",
                native_language="English",
            )
            result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_invalid_error_type_is_sanitized() -> None:
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "definitely_not_valid",
                "explanation": "Chat is masculine.",
            }
        ],
        "difficulty": "A1",
    }

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        with patch("app.feedback.AsyncOpenAI") as mock_client:
            instance = mock_client.return_value
            instance.responses.create = AsyncMock(
                return_value=_mock_responses_api(mock_response)
            )

            request = FeedbackRequest(
                sentence="La chat noir est sur le table.",
                target_language="French",
                native_language="English",
            )
            result = await get_feedback(request)

    assert result.errors[0].error_type == "other"


@pytest.mark.asyncio
async def test_cache_reuses_previous_result() -> None:
    mock_response = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "Location with 住む uses に.",
            }
        ],
        "difficulty": "A2",
    }

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        with patch("app.feedback.AsyncOpenAI") as mock_client:
            instance = mock_client.return_value
            instance.responses.create = AsyncMock(
                return_value=_mock_responses_api(mock_response)
            )

            request = FeedbackRequest(
                sentence="私は東京を住んでいます。",
                target_language="Japanese",
                native_language="English",
            )
            first = await get_feedback(request)
            second = await get_feedback(request)

    assert first == second
    assert instance.responses.create.await_count == 1


@pytest.mark.asyncio
async def test_missing_api_key_returns_clear_error() -> None:
    request = FeedbackRequest(
        sentence="Hola mundo",
        target_language="Spanish",
        native_language="English",
    )

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(Exception, match="OPENAI_API_KEY is not set"):
            await get_feedback(request)


@pytest.mark.asyncio
async def test_extract_json_handles_wrapper_text() -> None:
    payload = '{"corrected_sentence":"Hola","is_correct":true,"errors":[],"difficulty":"A1"}'
    wrapped = f"Here is the JSON you requested:\n{payload}"
    assert _extract_json(wrapped)["corrected_sentence"] == "Hola"


@pytest.mark.asyncio
async def test_changed_sentence_without_errors_creates_fallback_error() -> None:
    mock_response = {
        "corrected_sentence": "Eu quero comprar um presente.",
        "is_correct": False,
        "errors": [],
        "difficulty": "A1",
    }

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        with patch("app.feedback.AsyncOpenAI") as mock_client:
            instance = mock_client.return_value
            instance.responses.create = AsyncMock(
                return_value=_mock_responses_api(mock_response)
            )
            request = FeedbackRequest(
                sentence="Eu quero compra um presente.",
                target_language="Portuguese",
                native_language="English",
            )
            result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "other"
    assert "English" in result.errors[0].explanation


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure() -> None:
    mock_response = {
        "corrected_sentence": "Hola mundo",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        with patch("app.feedback.AsyncOpenAI") as mock_client, patch("app.feedback.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            instance = mock_client.return_value
            instance.responses.create = AsyncMock(side_effect=[Exception("temporary"), _mock_responses_api(mock_response)])

            request = FeedbackRequest(
                sentence="Hola mundo",
                target_language="Spanish",
                native_language="English",
            )
            result = await get_feedback(request)

    assert result.is_correct is True
    assert instance.responses.create.await_count == 2
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_whitespace_only_input_rejected_by_model_validation() -> None:
    with pytest.raises(Exception):
        FeedbackRequest(
            sentence="   ",
            target_language="Spanish",
            native_language="English",
        )


def test_api_returns_structured_error_for_missing_api_key() -> None:
    client = TestClient(app)
    with patch.dict("os.environ", {}, clear=True):
        response = client.post(
            "/feedback",
            json={
                "sentence": "Hola mundo",
                "target_language": "Spanish",
                "native_language": "English",
            },
        )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "missing_api_key"
    assert "request_id" in body["error"]
