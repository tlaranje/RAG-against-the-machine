from typing import TYPE_CHECKING
from student.utils import bar
import bm25s
import json
import os

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

    def search(self, prompt: str, k: int = 1) -> list[dict]:
        res, _ = self.indexer.bm25.retrieve(bm25s.tokenize(prompt), k=k)
        return [self.indexer.metadata[i] for i in res[0]]

    def search_dataset(self, data_path: str, k: int, save_dir: str) -> None:
        # 1. Load Index (Yellow)
        with bar(desc="Loading index", color="yellow") as pbar:
            self.indexer.load()
            pbar.update(1)

        # 2. Load Dataset (Blue)
        with bar(desc="Loading dataset", color="blue") as pbar:
            with open(data_path, "r") as fd:
                raw_data = json.load(fd)
            pbar.update(1)

        # 3. Parse Questions (Magenta)
        flat_data = [q for sublist in raw_data.values() for q in sublist]
        questions = []
        with bar(
            total=len(flat_data), desc="Parsing", color="magenta"
        ) as pbar:
            for d in flat_data:
                questions.append(UnansweredQuestion(
                    question_id=str(d['question_id']),
                    question=str(d['question'])
                ))
                pbar.update(1)

        rag = RagDataset(rag_questions=questions)

        # 4. Search (Green)
        results = []
        with bar(
            total=len(rag.rag_questions), desc="Searching", color="green"
        ) as pbar:
            for q in rag.rag_questions:
                sources_data = self.search(q.question, k=k)

                retrieved_sources = [s["source"] for s in sources_data]

                main_content = "\n\n".join(
                    s["content"] for s in sources_data
                )
                main_content = main_content[:2000]

                results.append(MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=retrieved_sources,
                    content=main_content
                ))
                pbar.update(1)

        # 5. Save (Cyan)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.basename(data_path)
        out_path = os.path.join(save_dir, file_name)

        with bar(desc="Saving", color="cyan") as pbar:
            s_res = StudentSearchResults(search_results=results, k=k)
            with open(out_path, "w") as fd:
                json.dump(s_res.model_dump(), fd, indent=4)
            pbar.update(1)
