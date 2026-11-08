"""Shared request, parsed-document, citation, and answer data structures."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ContentType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ParsedElement(BaseModel):
    """One text, table, image, or chart element produced from a document page."""

    doc_id: str
    page: int = Field(ge=1)
    content_type: ContentType = ContentType.UNKNOWN
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    source_kind: str = "text"


class SourcePage(BaseModel):
    """A page cited as a source for an answer."""

    doc_id: str
    page: int = Field(ge=1)


class Evidence(BaseModel):
    """An answer excerpt with optional location on the source page."""

    doc_id: str
    page: int = Field(ge=1)
    text: str
    bbox: tuple[float, float, float, float] | None = None


class Citation(BaseModel):
    """A checked source attached to one answer claim."""

    doc_id: str
    page: int = Field(ge=1)
    excerpt: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    source_kind: str


class AnswerPart(BaseModel):
    """One answer claim and its checked citations."""

    text: str
    status: str = "unverified"
    citations: list[Citation] = Field(default_factory=list)


class Answer(BaseModel):
    """Response shape shared by baseline and multimodal readers."""

    answer: str
    source_pages: list[SourcePage] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    answer_parts: list[AnswerPart] = Field(default_factory=list)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list)
    mode: str = "hybrid"
    top_k: int = Field(default=3, ge=1, le=5)


class RetrievalMetadata(BaseModel):
    mode: str
    pages: int
    latency_ms: int
    source_pages: list[SourcePage] = Field(default_factory=list)


class ChatTurnRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=36)
    question: str = Field(min_length=1, max_length=4000)
    mode: str = "hybrid"


class ChatTurnResponse(BaseModel):
    id: str
    request_id: str
    document_id: str
    question: str
    answer: str
    mode: str
    source_pages: list[SourcePage] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    answer_parts: list[AnswerPart] = Field(default_factory=list)
    retrieval: RetrievalMetadata
    created_at: str | None = None


class PinRequest(BaseModel):
    turn_id: str = Field(min_length=1, max_length=36)


class PinnedAnswerResponse(BaseModel):
    id: str
    turn_id: str
    document_id: str
    question: str
    answer: str
    source_pages: list[SourcePage] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    answer_parts: list[AnswerPart] = Field(default_factory=list)
    created_at: str | None = None
