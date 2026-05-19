from pydantic import BaseModel, Field
from typing import List, Optional
import uuid


class MinimalSource(BaseModel):
    """A document chunk with its location and optional content."""

    file_path: str
    first_character_index: int
    last_character_index: int
    content: Optional[str] = None


class UnansweredQuestion(BaseModel):
    """A question without a ground-truth answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Collection of answered and unanswered RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results retrieved for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results extended with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Batch of search results produced by a student retriever."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Batch of search results that also include generated answers."""

    search_results: List[MinimalAnswer]
