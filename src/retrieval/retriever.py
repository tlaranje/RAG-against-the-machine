from src.ingestion import Indexer
from typing import TYPE_CHECKING
import bm25s

if TYPE_CHECKING:
    from src.models import MinimalSource


class Retriever:
    def __init__(self, indexer: Indexer) -> None:
        self.bm25: bm25s.BM25 = indexer.bm25
        self.metadata = indexer.metadata

    def search(self, prompt: str, k: int = 1) -> list["MinimalSource"]:
        prompt_tokens = bm25s.tokenize(prompt)
        results, scores = self.bm25.retrieve(prompt_tokens, k=k)
        return [self.metadata[i] for i in results[0]]

    def search_dataset(self) -> None:
        pass
