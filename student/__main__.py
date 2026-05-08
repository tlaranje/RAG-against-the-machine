from student.ingestion import Indexer, Parser
from student.evaluation import Evaluator
from student.retrieval import Retriever
from typing import TYPE_CHECKING
from rich import print
import traceback
import fire

if TYPE_CHECKING:
    from student.models import MinimalSource

SAVE_DIRR = "data/output/search_results"


class Main:
    def __init__(self) -> None:
        self.parser = Parser()
        self.indexer = Indexer(self.parser)
        self.retriever = Retriever(self.indexer)

    def index(self, max_chunk_size: int = 2000) -> None:
        self.indexer.index(max_chunk_size=max_chunk_size)
        print(
            "[cyan]Ingestion complete! Indices saved "
            "under data/processed/[/cyan]"
        )

    def search_dataset(
        self, dataset_path: str, k: int = 1, save_directory: str = SAVE_DIRR
    ) -> None:
        self.retriever.search_dataset(dataset_path, k, save_directory)

    def search(self, prompt: str, k: int = 1) -> list["MinimalSource"]:
        results = self.retriever.search(prompt, k=k)
        return results

    def answer(self, prompt: str, k: int = 1) -> None:
        results = self.search(prompt, k=k)
        print(results)

    def evaluate(
        self, student_answer_path: str, dataset_path: str, k: int = 1
    ) -> None:
        ks = [1, 3, 5, 10]
        evaluation = Evaluator()
        for ki in ks:
            recall = evaluation.evaluate(student_answer_path, dataset_path, ki)
            print(f"Recall@{ki}: {recall:.3f}")


if __name__ == "__main__":
    try:
        fire.Fire(Main)
    except Exception as e:
        traceback.print_exc()
        print(f"[bold red]Error:\n    {e}[/bold red]")
