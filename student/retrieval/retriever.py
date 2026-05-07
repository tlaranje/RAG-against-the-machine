from student.models import (
    RagDataset, UnansweredQuestion, MinimalSearchResults, StudentSearchResults
)
from typing import TYPE_CHECKING
import bm25s
import json
import os

if TYPE_CHECKING:
    from student.ingestion import Indexer
    from src.models import MinimalSource


class Retriever:
    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def search(self, prompt: str, k: int = 1) -> list["MinimalSource"]:
        prompt_tokens = bm25s.tokenize(prompt)
        results, scores = self.indexer.bm25.retrieve(prompt_tokens, k=k)
        return [self.indexer.metadata[i] for i in results[0]]

    def search_dataset(
        self, data_path: str, k: int, save_dirr: str
    ) -> None:
        self.indexer.load()

        try:
            questions: list[UnansweredQuestion] = []

            with open(data_path, "r") as fd:
                raw_data = json.load(fd)

            for question_list in raw_data.values():
                for data in question_list:
                    questions.append(UnansweredQuestion(
                        question_id=str(data['question_id']),
                        question=str(data['question'])
                    ))

            rag = RagDataset(rag_questions=questions)
            search_results: list = []

            for curr_q in rag.rag_questions:
                sources = self.search(curr_q.question, k=k)

                search_results.append(MinimalSearchResults(
                    question_id=curr_q.question_id,
                    question=curr_q.question,
                    retrieved_sources=sources
                ))

            s_res = StudentSearchResults(search_results=search_results, k=k)

            os.makedirs(save_dirr.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(save_dirr, exist_ok=True)

            file_name = data_path.rsplit('/', 1)[1]
            with open(save_dirr + "/" + file_name, "w") as fd:
                json.dump(s_res.model_dump(), fd, indent=4)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in '{data_path}': {e}")
        except Exception as e:
            raise e
