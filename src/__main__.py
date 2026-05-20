import traceback

import fire
from rich import print
from src.evaluation import Evaluator
from src.generation import Generator
from src.ingestion import Indexer, Parser
from src.retrieval import Retriever

SAVE_DIRR = "data/output/"


class Main:
    """The main Command-Line Interface (CLI) entry point for the RAG system."""

    def __init__(self) -> None:
        """Initializes components needed for parsing, indexing, and search."""
        self.parser = Parser()
        self.indexer = Indexer(self.parser)
        self.retriever = Retriever(self.indexer)

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Indexes all documents and source code within the raw repository.

        Args:
            max_chunk_size: Maximum character length constraint for chunks.
        """
        # Python Fire sometimes reads numeric parameters as boolean flags
        # if input parsing is ambiguous. This safe-check restores the default.
        if isinstance(max_chunk_size, bool):
            max_chunk_size = 2000

        self.indexer.index(max_chunk_size=max_chunk_size)
        print(
            "\n[cyan]Ingestion complete! Indices saved "
            "under data/processed/[/cyan]\n"
        )

    def search_dataset(
        self, dataset_path: str, k: int = 1, save_directory: str = SAVE_DIRR
    ) -> None:
        """
        Queries the retrieval database using questions from a dataset file.

        Args:
            dataset_path: Path to the target evaluation JSON dataset file.
            k: Max amount of matched document snippets to look up.
            save_directory: Directory where the output JSON will be written.

        Raises:
            ValueError: If the k parameter is non-positive.
        """
        if k <= 0:
            raise ValueError("K must be positive int!")
        if isinstance(k, int):
            self.retriever.search_dataset(dataset_path, k, save_directory)

    def search(self, prompt: str, k: int = 1) -> None:
        """
        Performs a direct query search and prints matched code chunks.

        Args:
            prompt: Text string containing user search terms.
            k: Max amount of matched documents to show.
        """
        # Memory-map the indices to perform lightning-fast searches.
        self.indexer.load()
        results = self.retriever.search(prompt, k=k)
        print(results)

    def answer(self, prompt: str, k: int = 10) -> None:
        """Answers a single query by fetching context and feeding it to LLM.

        Args:
            prompt: Question asked by the user.
            k: Number of reference contexts to grab.
        """
        self.indexer.load()

        # Step 1: Search the DB for raw text or code snippets
        sources = self.retriever.search(prompt, k=k)

        # Step 2: Extract text from chunks, ignoring any empty contents
        context = "\n".join(
            s["source"]["content"] for s in sources if s.get("content")
        )

        # Step 3: Initialize Generator and query the local model
        generator = Generator()
        generator.answer(
            question=prompt,
            context=context,
        )

    def answer_dataset(
        self,
        src_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        """
        Generates structured answers for a file of search outputs.

        Args:
            src_search_results_path: Path to generated search results.
            save_directory: Directory where output JSON answers are saved.
        """
        generator = Generator()
        generator.answer_dataset(src_search_results_path, save_directory)

    def evaluate(
        self, src_answer_path: str, dataset_path: str, k: int = 1
    ) -> None:
        """
        Evaluates retrieval quality against ground-truth labels.

        Args:
            src_answer_path: Path to the src search results file.
            dataset_path: Path to ground-truth answers dataset JSON.
            k: Evaluates a single specific recall target parameter.
        """
        # Evaluates typical search target checkpoints for detailed reporting.
        ks = [1, 3, 5, 10]
        evaluation = Evaluator()
        for ki in ks:
            recall = evaluation.evaluate(src_answer_path, dataset_path, ki)
            print(f"Recall@{ki}: {recall:.3f}")


if __name__ == "__main__":
    try:
        # Launching CLI execution using Python Fire.
        fire.Fire(Main)
    except Exception as e:
        # Ensure we capture traceback information safely on standard error
        # to avoid silent failures or unreadable CLI application crashes.
        traceback.print_exc()
        print(f"[bold red]Error:\n    {e}[/bold red]")
