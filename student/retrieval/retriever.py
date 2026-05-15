from typing import TYPE_CHECKING
from student.utils import bar
import bm25s
import json
from student.models import (
    MinimalSearchResults, RagDataset, StudentSearchResults, UnansweredQuestion
)

if TYPE_CHECKING:
    from student.ingestion import Indexer


class Retriever:
    """Retriever using BM25 over the indexed knowledge base."""

    def __init__(self, indexer: "Indexer") -> None:
        self.indexer = indexer

    def search(self, prompt: str, k: int = 1) -> list[dict]:
        """Return top-k chunks for a given query.

        Args:
            prompt: The search query.
            k: Number of results to return.

        Returns:
            List of chunk dicts with 'source' and 'content' keys.
        """
        results, _ = self.indexer.bm25.retrieve(
            bm25s.tokenize(prompt), k=min(k, len(self.indexer.metadata))
        )
        return [self.indexer.metadata[i] for i in results[0]]

    def build_context(self, chunks: list[dict], max_chars: int = 1800) -> str:
        """
        Concatenate chunk contents up to max_chars to stay within token limits.

        Putting the most relevant chunk first so the LLM sees it early.

        Args:
            chunks: List of retrieved chunks ordered by relevance.
            max_chars: Maximum total characters for the context string.

        Returns:
            A single context string ready to be passed to the LLM.
        """
        parts = []
        total = 0
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                continue
            if total + len(content) > max_chars:
                # Fit whatever still fits
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(content[:remaining])
                break
            parts.append(content)
            total += len(content)
        return "\n---\n".join(parts)

    def search_dataset(
        self, data_path: str, k: int,
        save_dir: str, max_context_chars: int = 1800,
    ) -> None:
        """Run retrieval over a full question dataset and save results.

        Args:
            data_path: Path to the JSON dataset file.
            k: Number of sources to retrieve per question.
            save_dir: Path where the output JSON will be written.
            max_context_chars: Max characters of context passed to generator.
        """
        # 1. Load Index
        with bar(desc="Loading index", color="yellow") as pbar:
            self.indexer.load()
            pbar.update(1)

        # 2. Load Dataset
        with bar(desc="Loading dataset", color="blue") as pbar:
            with open(data_path, "r") as fd:
                raw_data = json.load(fd)
            pbar.update(1)

        # 3. Parse with pydantic — handles both AnsweredQuestion and
        #    UnansweredQuestion transparently
        with bar(desc="Parsing", color="magenta") as pbar:
            rag = RagDataset.model_validate(raw_data)
            questions: list[UnansweredQuestion] = [
                UnansweredQuestion(
                    question_id=str(q.question_id),
                    question=str(q.question),
                )
                for q in rag.rag_questions
            ]
            pbar.update(1)

        # 4. Search — keep ALL k sources, build merged context
        results: list[MinimalSearchResults] = []
        with bar(
            total=len(questions), desc="Searching", color="green"
        ) as pbar:
            for q in questions:
                chunks = self.search(q.question, k=k)

                if chunks:
                    # All k sources for recall@k evaluation
                    retrieved_sources = [
                        chunk["source"] for chunk in chunks
                    ]
                    # Merged context for the generator
                    content = self.build_context(
                        chunks, max_chars=max_context_chars
                    )
                else:
                    retrieved_sources = []
                    content = ""

                results.append(MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=retrieved_sources,
                    content=content,
                ))
                pbar.update(1)

        # 5. Save
        with bar(desc="Saving", color="cyan") as pbar:
            s_res = StudentSearchResults(search_results=results, k=k)
            with open(save_dir, "w") as fd:
                json.dump(s_res.model_dump(), fd, indent=4, ensure_ascii=False)
            pbar.update(1)
