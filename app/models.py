from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}

VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}

MAX_SENTENCE_LENGTH = 2000
MAX_LANGUAGE_LENGTH = 80


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ErrorDetail(StrictBaseModel):
    original: str = Field(
        min_length=1,
        max_length=500,
        description="The erroneous word or phrase from the original sentence",
    )
    correction: str = Field(
        min_length=1,
        max_length=500,
        description="The corrected word or phrase",
    )
    error_type: Literal[
        "grammar",
        "spelling",
        "word_choice",
        "punctuation",
        "word_order",
        "missing_word",
        "extra_word",
        "conjugation",
        "gender_agreement",
        "number_agreement",
        "tone_register",
        "other",
    ] = Field(description="Category of the error")
    explanation: str = Field(
        min_length=1,
        max_length=600,
        description="A brief, learner-friendly explanation written in the native language",
    )


class FeedbackRequest(StrictBaseModel):
    sentence: str = Field(
        min_length=1,
        max_length=MAX_SENTENCE_LENGTH,
        description="The learner's sentence in the target language",
    )
    target_language: str = Field(
        min_length=2,
        max_length=MAX_LANGUAGE_LENGTH,
        description="The language the learner is studying",
    )
    native_language: str = Field(
        min_length=2,
        max_length=MAX_LANGUAGE_LENGTH,
        description="The learner's native language -- explanations will be in this language",
    )

    @field_validator("sentence", "target_language", "native_language")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Value must not be blank")
        return value.strip()


class FeedbackResponse(StrictBaseModel):
    corrected_sentence: str = Field(
        min_length=1,
        max_length=MAX_SENTENCE_LENGTH,
        description="The grammatically corrected version of the input sentence",
    )
    is_correct: bool = Field(description="true if the original sentence had no errors")
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of errors found. Empty if the sentence is correct.",
    )
    difficulty: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="CEFR difficulty level: A1, A2, B1, B2, C1, or C2"
    )
