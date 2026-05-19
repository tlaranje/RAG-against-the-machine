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
        Initialise the Retriever with a pre-built indexer.

        Args:
            indexer: Indexer instance holding the BM25 model and metadata.
        """
        self.indexer = indexer

    def search(self, prompt: str, k: int = 1) -> list[dict]:
        """
        Retrieve the top-k chunks most relevant to a prompt.

        Args:
            prompt: Query string to search for.
            k: Number of chunks to retrieve.

        Returns:
            List of chunk dicts.
        """
        results, _ = self.indexer.bm25.retrieve(
            bm25s.tokenize(
                prompt,
                stopwords="en",
                stemmer=self.indexer.stemmer.stemWords,
            ),
            k=min(k, len(self.indexer.metadata))
        )

        return [self.indexer.metadata[idx].copy() for idx in results[0]]

    def build_context(self, chunks: list[dict], max_chars: int = 1800) -> str:
        """
        Concatenate chunk contents up to a character budget.

        Args:
            chunks: List of chunk dicts containing a ``source.content`` key.
            max_chars: Maximum total characters in the returned context.

        Returns:
            Chunks joined by ``\\n---\\n``, truncated to ``max_chars``.
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
        Run retrieval over a full dataset and save the results.

        Args:
            data_path: Path to the JSON file with the ``RagDataset``.
            k: Number of sources to retrieve per question.
            save_dir: Output path for the JSON results file.
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
        #    UnansweredQuestion transparently.
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

        # 4. Search — keep ALL k sources and build merged context.
        results: list[MinimalSearchResults] = []
        with bar(
            total=len(questions), desc="Searching", color="green"
        ) as pbar:
            for q in questions:
                chunks = self.search(q.question, k=k)
                if chunks:
                    retrieved_sources = [
                        MinimalSource(
                            file_path=chunk["source"]["file_path"],
                            first_character_index=chunk["source"][
                                "first_character_index"
                            ],
                            last_character_index=chunk["source"][
                                "last_character_index"
                            ],
                            content=chunk["source"]["content"],
                        )
                        for chunk in chunks
                    ]
                else:
                    retrieved_sources = []
                results.append(MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=retrieved_sources
                ))
                pbar.update(1)

        # 5. Save results to disk as JSON.
        if save_dir.endswith("/") or os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, "dataset.json")
        else:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            file_path = save_dir

        with bar(desc="Saving", color="cyan") as pbar:
            s_res = StudentSearchResults(search_results=results, k=k)
            data = s_res.model_dump()

            for result in data["search_results"]:
                result["question_str"] = result.pop("question")

            with open(file_path, "w") as fd:
                json.dump(data, fd, indent=4, ensure_ascii=False)

            pbar.update(1)
