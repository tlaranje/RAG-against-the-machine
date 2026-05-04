from src.ingestion import Chunker, Indexer, Parser
from src.retrieval import Retriever
from typing import TYPE_CHECKING
from rich import print
import fire

if TYPE_CHECKING:
    from src.models import MinimalSource


class Student:
    def index(self, max_chunk_size: int = 2000) -> None:
        parser = Parser()
        files = parser.parse_directory("data/raw/vllm-0.10.1")
        chunker = Chunker(max_chunk_size=max_chunk_size)
        chunks = {}
        for file_path, content in files.items():
            chunks.update(chunker.chunk_file({file_path: content}))
        indexer = Indexer()
        indexer.index(chunks)
        indexer.save()
        print(
            "[cyan]Ingestion complete! Indices saved "
            "under data/processed/[/cyan]"
        )

    def search(self, prompt: str, k: int = 1) -> list["MinimalSource"]:
        indexer = Indexer()
        indexer.load()
        retriever = Retriever(indexer)
        results = retriever.search(prompt, k=k)
        print(results)
        return results

    def answer(self, prompt: str, k: int = 10) -> None:
        results = self.search(prompt, k=k)
        print(results)


if __name__ == "__main__":
    fire.Fire(Student)
