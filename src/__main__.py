import os
from rich import print
from rich.panel import Panel
from rich.columns import Columns
from src.chunkers import Chunker


def test_chunking():
    max_size = 200
    chunker = Chunker(max_chunk_size=max_size)

    python_code = """
        class Database:
            def connect(self):
                print("Connecting...")

        def main():
            db = Database()
            db.connect()
            if True:
                print("Success")
    """

    markdown_text = """
        # Project Overview
        This is a RAG system. It uses BM25 for retrieval.
        The goal is to answer questions about vLLM.

        ## Features
        - Fast indexing.
        - Accurate retrieval.
        - Intelligent chunking.
    """

    print(Panel("[bold blue]Testing Python Code Chunking[/bold blue]"))
    py_chunks = chunker.chunk_file("test.py", python_code)
    for i, chunk in enumerate(py_chunks):
        print(Panel(
            chunk,
            title=f"Python Chunk {i+1}",
            border_style="green",
            expand=False)
        )

    print("\n" + "="*50 + "\n")

    print(Panel("[bold magenta]Testing Text/Markdown Chunking[/bold magenta]"))
    text_chunks = chunker.chunk_file("readme.md", markdown_text)
    for i, chunk in enumerate(text_chunks):
        print(Panel(
            chunk,
            title=f"Text Chunk {i+1}",
            border_style="cyan",
            expand=False)
        )

    print(f"\n[bold]Configured Max Size:[/bold] {max_size}")
    print(f"[bold]Python Chunks generated:[/bold] {len(py_chunks)}")
    print(f"[bold]Text Chunks generated:[/bold] {len(text_chunks)}")


if __name__ == "__main__":
    test_chunking()
