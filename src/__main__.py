from src.ingestion import Indexer, Parser
from src.generation import Generator
from src.models import MinimalSource
from src.retrieval import Retriever
from rich import print
import fire

DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
SAVE_DIRR = "data/output/search_results"
SAVE_DIRR_ANSWERS = "data/output/search_results_and_answer"
K = 10
MAX_CHUNK_SIZE = 2000
PROMPT = "How to configure OpenAI server?"
STUDENT_SEARCH_RESULTS_PATH = (
    "data/output/search_results/dataset_docs_public.json"
)


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
        max_chunk_size = (
            MAX_CHUNK_SIZE
            if isinstance(max_chunk_size, bool) else max_chunk_size
        )

        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive int!")

        self.indexer.index(max_chunk_size=max_chunk_size)

        print(
            "[cyan]Ingestion complete! Indices saved "
            "under data/processed/[/cyan]"
        )

    def search(self, prompt: str = PROMPT, k: int = K) -> None:
        """
        Performs a direct query search and prints matched code chunks.

        Args:
            prompt: Text string containing user search terms.
            k: Max amount of matched documents to show.
        """
        self.indexer.load()

        prompt = (PROMPT if isinstance(prompt, bool) else prompt)
        k = (K if isinstance(k, bool) else k)

        results = self.retriever.search(prompt, k=k)
        print(results)

    def search_dataset(
        self, dataset_path: str = DATASET_PATH,
        k: int = K, save_directory: str = SAVE_DIRR
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
        dataset_path = (
            DATASET_PATH if isinstance(dataset_path, bool) else dataset_path
        )
        k = (K if isinstance(k, bool) else k)
        save_directory = (
            SAVE_DIRR if isinstance(save_directory, bool) else save_directory
        )

        if k <= 0:
            raise ValueError("K must be positive int!")

        self.retriever.search_dataset(dataset_path, k, save_directory)

    def answer(self, prompt: str = PROMPT, k: int = K) -> None:
        """
        Answers a single query by fetching context and feeding it to LLM.

        Args:
            prompt: Question asked by the user.
            k: Number of reference contexts to grab.
        """
        self.indexer.load()
        prompt = (PROMPT if isinstance(prompt, bool) else prompt)
        k = (K if isinstance(k, bool) else k)

        # Step 1: Search the DB for raw text or code snippets
        sources = self.retriever.search(prompt, k=k)

        # Step 2: Build MinimalSource objects directly from search results
        minimal_sources = [
            MinimalSource(
                file_path=s.get("file_path", "unknown"),
                first_character_index=s.get("first_character_index", 0),
                last_character_index=s.get("last_character_index", 0),
                content=(
                    s.get("content") or s.get("source", {}).get("content", "")
                ),
            )
            for s in sources
        ]

        # Step 3: Initialize Generator and query the local model
        generator = Generator()
        generator.answer(
            question=prompt,
            sources=minimal_sources,
        )

    def answer_dataset(
        self, student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        save_directory: str = SAVE_DIRR_ANSWERS
    ) -> None:
        """
        Generates structured answers for a file of search outputs.

        Args:
            src_search_results_path: Path to generated search results.
            save_directory: Directory where output JSON answers are saved.
        """
        student_search_results_path = (
            STUDENT_SEARCH_RESULTS_PATH
            if isinstance(student_search_results_path, bool) else
            student_search_results_path
        )
        save_directory = (
            SAVE_DIRR_ANSWERS if isinstance(save_directory, bool) else
            save_directory
        )
        generator = Generator()
        generator.answer_dataset(student_search_results_path, save_directory)


if __name__ == "__main__":
    try:
        # Launching CLI execution using Python Fire.
        fire.Fire(Main)
    except Exception as e:
        # Ensure we capture traceback information safely on standard error
        # to avoid silent failures or unreadable CLI application crashes.
        print(f"[bold red]Error:\n    {e}[/bold red]")
