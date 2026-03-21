from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class ErrorDetail(BaseModel):
    original: str = Field(
        description="The erroneous word or phrase from the original sentence"
    )
    correction: str = Field(description="The corrected word or phrase")
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
        description="A brief, learner-friendly explanation written in the native language"
    )


class FeedbackRequest(BaseModel):
    sentence: str = Field(
        min_length=1, description="The learner's sentence in the target language"
    )
    target_language: str = Field(
        min_length=2, description="The language the learner is studying"
    )
    native_language: str = Field(
        min_length=2,
        description="The learner's native language -- explanations will be in this language",
    )


class FeedbackResponse(BaseModel):
    corrected_sentence: str = Field(
        description="The grammatically corrected version of the input sentence"
    )
    is_correct: bool = Field(description="true if the original sentence had no errors")
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of errors found. Empty if the sentence is correct.",
    )
    difficulty: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="CEFR difficulty level: A1, A2, B1, B2, C1, or C2"
    )
