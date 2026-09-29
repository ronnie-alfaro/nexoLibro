"""Bounded chat requests and traceable retrieved evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=6)
    document_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Escribe una pregunta")
        return value


class Source(BaseModel):
    number: int = 0
    point_id: str
    document_id: str
    title: str
    author: str | None = None
    filename: str
    section: str | None = None
    page_number: int | None = None
    chunk_index: int
    score: float
    text: str
