from student.ingestion import Indexer, Parser
from student.evaluation import Evaluator
from student.retrieval import Retriever
from student.generation import Generator
from rich import print
import traceback
import fire

SAVE_DIRR = "data/output/"


class Main:
    def __init__(self) -> None:
        self.parser = Parser()
        self.indexer = Indexer(self.parser)
        self.retriever = Retriever(self.indexer)

    def index(self, max_chunk_size: int = 2000) -> None:
        self.indexer.index(max_chunk_size=max_chunk_size)
        print(
            "\n[cyan]Ingestion complete! Indices saved "
            "under data/processed/[/cyan]\n"
        )

    def search_dataset(
        self, dataset_path: str, k: int = 1, save_directory: str = SAVE_DIRR
    ) -> None:
        self.retriever.search_dataset(dataset_path, k, save_directory)

    def search(self, prompt: str, k: int = 1) -> None:
        self.indexer.load()
        results = self.retriever.search(prompt, k=k)
        print(results)

    def answer(self, prompt: str, k: int = 10) -> None:
        self.indexer.load()

        sources = self.retriever.search(prompt, k=k)

        # extrai só o texto de cada chunk
        context = "\n".join(
           s["source"]["content"] for s in sources if s.get("content")
        )

        generator = Generator()

        generator.answer(
            question=prompt,
            context=context,
        )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        generator = Generator()
        generator.answer_dataset(student_search_results_path, save_directory)

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
