from typing import TYPE_CHECKING, Any
from src.utils import bar
from pathlib import Path
from rich import print
import json
import os

from src.models import (
    MinimalSearchResults, MinimalSource, RagDataset, StudentSearchResults
)

if TYPE_CHECKING:
    from src.ingestion import Indexer


class Retriever:
    """Retrieves snippets from documents or source code using pure BM25."""

    def __init__(self, indexer: "Indexer") -> None:
        """Initializes the retriever engine with a specific dataset indexer."""
        self.indexer = indexer

    def _is_valid_file(self, file_path: str, is_code_index: bool) -> bool:
        """Validates if a file path fits the requested target index type."""
        ext = Path(file_path).suffix.lower()
        if is_code_index:
            return ext == ".py"
        return ext in {".md", ".txt"}

    def _search_index(
        self, query: str, k: int, is_code_index: bool
    ) -> list[dict[str, Any]]:
        bm25 = self.indexer.bm25
        metadata = self.indexer.metadata_all
        tokenized = self.indexer.tokenizer([query])

        fetch_k = min(k * 2, len(metadata))
        results, scores = bm25.retrieve(tokenized, k=fetch_k)

        valid_chunks = []
        for i, idx in enumerate(results[0]):
            chunk = metadata[int(idx)]
            file_path = chunk.get("source", {}).get("file_path", "")

            if self._is_valid_file(file_path, is_code_index):
                chunk_copy = chunk.copy()
                chunk_copy["_score"] = float(scores[0][i])
                valid_chunks.append(chunk_copy)

            if len(valid_chunks) >= k:
                break

        return valid_chunks

    def search(
        self, prompt: str, k: int, index_type: str = "both"
    ) -> list[dict[str, Any]]:
        """Executes search on code, docs, or both indices using pure BM25."""
        all_results: list[dict[str, Any]] = []

        if index_type in ("code", "both"):
            all_results.extend(
                self._search_index(prompt, k, is_code_index=True)
            )
        if index_type in ("docs", "both"):
            all_results.extend(
                self._search_index(prompt, k, is_code_index=False)
            )

        # Ordena puramente pelo score retornado pelo BM25
        all_results = sorted(
            all_results, key=lambda x: x["_score"], reverse=True
        )
        return all_results[:k]

    def build_context(
        self, chunks: list[dict[str, Any]], max_chars: int = 4000
    ) -> str:
        parts = []
        total = 0
        for chunk in chunks:
            content = chunk.get("source", {}).get("content", "").strip()
            if not content:
                continue
            if total + len(content) > max_chars:
                break
            parts.append(content)
            total += len(content)
        return "\n---\n".join(parts)

    def search_dataset(self, data_path: str, k: int, save_dir: str) -> None:
        with bar(desc="Loading index", color="yellow") as pbar:
            self.indexer.load()
            pbar.update(1)

        dataset_name = Path(data_path).stem.lower()
        if "code" in dataset_name:
            index_type = "code"
        elif "docs" in dataset_name:
            index_type = "docs"
        else:
            index_type = "both"

        with open(data_path, "r", encoding="utf-8") as fd:
            raw_data = json.load(fd)
            rag = RagDataset.model_validate(raw_data)

        results = []
        for q in bar(rag.rag_questions, desc="Searching"):
            chunks = self.search(q.question, k=k, index_type=index_type)
            retrieved_sources = [
                MinimalSource(
                    file_path=chunk["source"]["file_path"],
                    first_character_index=chunk[
                        "source"]["first_character_index"],
                    last_character_index=chunk[
                        "source"]["last_character_index"],
                    content=chunk["source"]["content"],
                )
                for chunk in chunks
            ]
            results.append(
                MinimalSearchResults(
                    question_id=str(q.question_id),
                    question_str=q.question,
                    retrieved_sources=retrieved_sources,
                )
            )

        if save_dir.endswith("/") or (
            not save_dir.endswith("/") and not save_dir.endswith(".json")
        ):
            os.makedirs(save_dir, exist_ok=True)
            file_name = data_path.rsplit("/", 1)[-1]
            file_path = os.path.join(save_dir, file_name)
        else:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            file_path = save_dir

        print(f"[bold green]File saved on {file_path} [/bold green]")
        with bar(desc="Saving", color="cyan") as pbar:
            s_res = StudentSearchResults(search_results=results, k=k)
            with open(file_path, "w", encoding="utf-8") as fd:
                json.dump(s_res.model_dump(), fd, indent=4, ensure_ascii=False)
            pbar.update(1)
