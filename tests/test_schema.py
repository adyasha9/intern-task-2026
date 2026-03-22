import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_DIR = Path(__file__).parent.parent / "schema"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def load_examples() -> list[dict]:
    return json.loads((EXAMPLES_DIR / "sample_inputs.json").read_text())


class TestRequestSchema:
    def test_valid_request(self) -> None:
        schema = load_schema("request.schema.json")
        valid = {
            "sentence": "Hola mundo",
            "target_language": "Spanish",
            "native_language": "English",
        }
        jsonschema.validate(valid, schema)

    def test_missing_sentence_fails(self) -> None:
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"target_language": "Spanish", "native_language": "English"},
                schema,
            )

    def test_blank_sentence_fails(self) -> None:
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"sentence": "   ", "target_language": "Spanish", "native_language": "English"},
                schema,
            )

    def test_request_string_length_limits_are_enforced(self) -> None:
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "sentence": "x" * 2001,
                    "target_language": "Spanish",
                    "native_language": "English",
                },
                schema,
            )

    def test_extra_request_field_fails(self) -> None:
        schema = load_schema("request.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "sentence": "Hola",
                    "target_language": "Spanish",
                    "native_language": "English",
                    "unexpected": True,
                },
                schema,
            )


class TestResponseSchema:
    def test_correct_response(self) -> None:
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "Hola mundo",
            "is_correct": True,
            "errors": [],
            "difficulty": "A1",
        }
        jsonschema.validate(valid, schema)

    def test_response_with_errors(self) -> None:
        schema = load_schema("response.schema.json")
        valid = {
            "corrected_sentence": "Le chat noir",
            "is_correct": False,
            "errors": [
                {
                    "original": "La chat",
                    "correction": "Le chat",
                    "error_type": "gender_agreement",
                    "explanation": "Chat is masculine",
                }
            ],
            "difficulty": "A1",
        }
        jsonschema.validate(valid, schema)

    def test_invalid_difficulty_fails(self) -> None:
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "test",
                    "is_correct": True,
                    "errors": [],
                    "difficulty": "Z9",
                },
                schema,
            )

    def test_invalid_error_type_fails(self) -> None:
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "Hola mundo",
                    "is_correct": False,
                    "errors": [
                        {
                            "original": "Hola",
                            "correction": "Hola",
                            "error_type": "invalid",
                            "explanation": "Explanation",
                        }
                    ],
                    "difficulty": "A1",
                },
                schema,
            )

    def test_extra_error_field_fails(self) -> None:
        schema = load_schema("response.schema.json")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "corrected_sentence": "Hola mundo",
                    "is_correct": False,
                    "errors": [
                        {
                            "original": "Hola",
                            "correction": "Hola",
                            "error_type": "other",
                            "explanation": "Explanation",
                            "extra": "nope",
                        }
                    ],
                    "difficulty": "A1",
                },
                schema,
            )


class TestExamplesMatchSchemas:
    def test_all_example_requests_valid(self) -> None:
        schema = load_schema("request.schema.json")
        for example in load_examples():
            jsonschema.validate(example["request"], schema)

    def test_all_example_responses_valid(self) -> None:
        schema = load_schema("response.schema.json")
        for example in load_examples():
            jsonschema.validate(example["expected_response"], schema)
