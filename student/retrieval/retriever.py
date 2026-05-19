from typing import TYPE_CHECKING
from student.utils import bar
import bm25s
import json
import os
from student.models import (
    MinimalSearchResults, RagDataset, StudentSearchResults,
    UnansweredQuestion, MinimalSource
)

if TYPE_CHECKING:
    from student.ingestion import Indexer


class Retriever:
    def __init__(self, indexer: "Indexer") -> None:
        """
        Initialize retriever with a pre-built indexer.

        Args:
            indexer: Indexer containing BM25 model and metadata.
        """
        self.indexer = indexer

    def normalize(self, text: str) -> str:
        """
        Normalize text for BM25 tokenization.

        Args:
            text: Raw query string.

        Returns:
            Normalized string.
        """
        text = text.lower()
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        return text

    def search(self, prompt: str, k: int = 1) -> list[dict]:
        """
        Retrieve top-k chunks relevant to a query.

        Args:
            prompt: Query string.
            k: Number of chunks to retrieve.

        Returns:
            List of chunk dictionaries with BM25 scores.
        """
        tokens = bm25s.tokenize(
            self.normalize(prompt),
            stopwords="en",
            stemmer=self.indexer.stemmer.stemWords,
        )

        scores, indices = self.indexer.bm25.retrieve(
            tokens,
            k=min(k, len(self.indexer.metadata))
        )

        ranked = sorted(
            zip(indices[0], scores[0]),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                **self.indexer.metadata[int(idx)].copy(),
                "bm25_score": float(score)
            }
            for idx, score in ranked[:k]
        ]

    def build_context(self, chunks: list[dict], max_chars: int = 1800) -> str:
        """
        Build a context string from retrieved chunks.

        Args:
            chunks: Retrieved chunk dictionaries.
            max_chars: Maximum allowed characters.

        Returns:
            Concatenated context string.
        """
        parts = []
        total = 0

        for chunk in chunks:
            content = chunk.get("source", {}).get("content", "").strip()
            if not content:
                continue

            if total + len(content) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(content[:remaining])
                break

            parts.append(content)
            total += len(content)

        return "\n---\n".join(parts)

    def search_dataset(self, data_path: str, k: int, save_dir: str) -> None:
        """
        Run retrieval on a dataset and save results.

        Args:
            data_path: Path to dataset JSON.
            k: Number of chunks per question.
            save_dir: Output directory or file path.
        """
        # Load index
        with bar(desc="Loading index", color="yellow") as pbar:
            self.indexer.load()
            pbar.update(1)

        # Load dataset
        with bar(desc="Loading dataset", color="blue") as pbar:
            with open(data_path, "r") as fd:
                raw_data = json.load(fd)
            pbar.update(1)

        # Parse dataset
        with bar(desc="Parsing", color="magenta") as pbar:
            rag = RagDataset.model_validate(raw_data)
            questions = [
                UnansweredQuestion(
                    question_id=str(q.question_id),
                    question=str(q.question),
                )
                for q in rag.rag_questions
            ]
            pbar.update(1)

        # Perform retrieval
        results = []

        with bar(
            total=len(questions), desc="Searching", color="green"
        ) as pbar:
            for q in questions:
                chunks = self.search(q.question, k=k)

                if chunks:
                    retrieved_sources = [
                        MinimalSource(
                            file_path=c["source"]["file_path"],
                            first_character_index=c["source"][
                                "first_character_index"
                            ],
                            last_character_index=c["source"][
                                "last_character_index"
                            ],
                            content=c["source"]['content']
                        )
                        for c in chunks
                    ]
                else:
                    retrieved_sources = []

                results.append(
                    MinimalSearchResults(
                        question_id=q.question_id,
                        question_str=q.question,
                        retrieved_sources=retrieved_sources
                    )
                )
                pbar.update(1)

        # Prepare output path
        if save_dir.endswith("/") or os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, "dataset.json")
        else:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            file_path = save_dir

        # Save results
        with bar(desc="Saving", color="cyan") as pbar:
            s_res = StudentSearchResults(
                search_results=results,
                k=k
            )
            data = s_res.model_dump()

            with open(file_path, "w") as fd:
                json.dump(data, fd, indent=4, ensure_ascii=False)

            pbar.update(1)
