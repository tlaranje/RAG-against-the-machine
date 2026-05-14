from typing import TYPE_CHECKING
from student.utils import bar

import bm25s
import json
import os
import re

from student.models import (
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)

if TYPE_CHECKING:
    from student.ingestion import Indexer


class Retriever:

    def __init__(self, indexer: "Indexer") -> None:
        self.indexer = indexer

    # ---------------------------------------------------------
    # Fast normalization
    # ---------------------------------------------------------

    def normalize(self, text: str) -> str:

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s/_-]",
            " ",
            text
        )

        return re.sub(
            r"\s+",
            " ",
            text
        ).strip()

    # ---------------------------------------------------------
    # Better lexical reranking
    # ---------------------------------------------------------

    def overlap_score(
        self,
        question: str,
        content: str
    ) -> float:

        q_words = set(
            self.normalize(question).split()
        )

        c_words = set(
            self.normalize(content[:500]).split()
        )

        if not q_words:
            return 0.0

        overlap = len(q_words & c_words)

        # normalize by query size
        return overlap / len(q_words)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        prompt: str,
        k: int = 3
    ) -> list[dict]:

        normalized_prompt = self.normalize(prompt)

        tokens = bm25s.tokenize(normalized_prompt)

        # IMPORTANT:
        # smaller candidate pool = faster
        res, _ = self.indexer.bm25.retrieve(
            tokens,
            k=max(k * 2, 6)
        )

        candidates = [
            self.indexer.metadata[i]
            for i in res[0]
        ]

        ranked = sorted(
            candidates,
            key=lambda x: self.overlap_score(
                prompt,
                x["content"]
            ),
            reverse=True
        )

        return ranked[:k]

    # ---------------------------------------------------------
    # Dataset Search
    # ---------------------------------------------------------

    def search_dataset(
        self,
        data_path: str,
        k: int,
        save_dir: str,
    ) -> None:

        # -----------------------------------------------------
        # Load index
        # -----------------------------------------------------

        with bar(
            desc="Loading index",
            color="yellow"
        ) as pbar:

            self.indexer.load()

            pbar.update(1)

        # -----------------------------------------------------
        # Load dataset
        # -----------------------------------------------------

        with bar(
            desc="Loading dataset",
            color="blue"
        ) as pbar:

            with open(
                data_path,
                "r",
                encoding="utf-8"
            ) as fd:

                raw_data = json.load(fd)

            pbar.update(1)

        # -----------------------------------------------------
        # Parse questions
        # -----------------------------------------------------

        flat_data = [
            q
            for sublist in raw_data.values()
            for q in sublist
        ]

        questions = []

        with bar(
            total=len(flat_data),
            desc="Parsing",
            color="magenta"
        ) as pbar:

            for d in flat_data:

                questions.append(
                    UnansweredQuestion(
                        question_id=str(d["question_id"]),
                        question=str(d["question"])
                    )
                )

                pbar.update(1)

        rag = RagDataset(
            rag_questions=questions
        )

        # -----------------------------------------------------
        # Search
        # -----------------------------------------------------

        results = []

        with bar(
            total=len(rag.rag_questions),
            desc="Searching",
            color="green"
        ) as pbar:

            for q in rag.rag_questions:

                sources_data = self.search(
                    q.question,
                    k=k
                )

                retrieved_sources = [
                    s["source"]
                    for s in sources_data
                ]

                # IMPORTANT:
                # ONLY USE BEST CHUNK
                best_chunk = ""

                if sources_data:

                    best_chunk = (
                        sources_data[0]["content"]
                        .strip()[:500]
                    )

                results.append(
                    MinimalSearchResults(
                        question_id=q.question_id,
                        question=q.question,
                        retrieved_sources=retrieved_sources,
                        content=best_chunk
                    )
                )

                pbar.update(1)

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        file_name = os.path.basename(data_path)

        out_path = os.path.join(
            save_dir,
            file_name
        )

        with bar(
            desc="Saving",
            color="cyan"
        ) as pbar:

            s_res = StudentSearchResults(
                search_results=results,
                k=k
            )

            with open(
                out_path,
                "w",
                encoding="utf-8"
            ) as fd:

                json.dump(
                    s_res.model_dump(),
                    fd,
                    indent=4,
                    ensure_ascii=False
                )

            pbar.update(1)

        print(
            f"\n[bold green]Saved:[/bold green] {out_path}"
        )
