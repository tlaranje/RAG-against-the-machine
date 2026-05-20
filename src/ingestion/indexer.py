from typing import Any, TYPE_CHECKING
from src.ingestion import Chunker
from src.utils import bar
from Stemmer import Stemmer
import bm25s
import json
import os

if TYPE_CHECKING:
    from src.ingestion import Parser


def is_code_file(path: str) -> bool:
    """
    Checks if a file path points to a Python source file.

    Args:
        path: The file path string to evaluate.

    Returns:
        True if it is a python file, False otherwise.
    """
    p = path.lower().strip()
    # Strips trailing URL-like anchors, parameters, or line numbers.
    p = p.split(":")[0]
    p = p.split("#")[0]
    p = p.split("?")[0]
    return ".py" in p


class Indexer:
    """Manages chunking, tokenization, and BM25 index creation for RAG."""

    def __init__(
        self, parser: "Parser", base_path: str = "data/processed"
    ) -> None:
        """
        Initializes the Indexer engine with indices and metadata storage.

        Args:
            parser: Parser object used to scan and read raw contents.
            base_path: Storage destination folder for processed files.
        """
        self.parser = parser
        self.base_path = base_path

        # Dual-index architecture separating docs and code.
        self.bm25_docs = bm25s.BM25()
        self.bm25_code = bm25s.BM25()

        self.metadata_docs: list[dict[str, Any]] = []
        self.metadata_code: list[dict[str, Any]] = []

        self.stemmer = Stemmer("english")

        # Lambda tokenizers used during indexing and re-used by the retriever.
        # For docs: removes English stopwords and applies word stemming.
        self.tokenizer_docs: Any = lambda texts: bm25s.tokenize(
            texts, stopwords="en", stemmer=self.stemmer.stemWords
        )
        # For code: preserves variables/syntax exactly as they appear.
        self.tokenizer_code: Any = lambda texts: bm25s.tokenize(
            texts, stopwords=[], stemmer=lambda x: x
        )

        # File paths where artifacts will be saved or loaded.
        self.docs_index_path = f"{base_path}/bm25_index_docs"
        self.code_index_path = f"{base_path}/bm25_index_code"
        self.docs_metadata_path = f"{base_path}/chunks/metadata_docs.json"
        self.code_metadata_path = f"{base_path}/chunks/metadata_code.json"

    def index(self, max_chunk_size: int = 1500) -> None:
        """
        Indexes all text and code files found in the dataset folder.

        Args:
            max_chunk_size: Maximum character count limit for each text block.
        """
        files = self.parser.parse_directory("data/raw/vllm-0.10.1")
        chunker = Chunker(max_chunk_size=max_chunk_size)

        docs_texts, docs_sources = [], []
        code_texts, code_sources = [], []

        # Step 1: Split raw files into text chunks and sort by file type.
        with bar(total=len(files), desc="Chunking", color="green") as pbar:
            for path, content in files.items():
                chunks = chunker.chunk_file({path: content})

                for _, data in chunks.items():
                    text = data["text"]
                    source = data["source"]
                    file_path = source.file_path

                    if is_code_file(file_path):
                        # Prepend file path text so BM25 can easily match
                        # specific queries targeting class or module names.
                        enriched = f"{file_path}\n{text}"
                        code_texts.append(enriched)
                        code_sources.append(source)
                    else:
                        docs_texts.append(text)
                        docs_sources.append(source)

                pbar.update(1)

        # Step 2: Convert structural strings into token streams.
        with bar(desc="Tokenizing", color="blue") as pbar:
            docs_tokens: Any = self.tokenizer_docs(docs_texts)
            code_tokens: Any = self.tokenizer_code(code_texts)
            pbar.update(1)

        # Step 3: Fit the BM25 statistical matrix with the token datasets.
        with bar(desc="Building Indexes", color="yellow") as pbar:
            self.bm25_docs.index(docs_tokens)
            self.bm25_code.index(code_tokens)
            pbar.update(1)

        # Step 4: Serialize source positions into lightweight dictionaries.
        self.metadata_docs = [
            {"source": s.model_dump()} for s in docs_sources
        ]
        self.metadata_code = [
            {"source": s.model_dump()} for s in code_sources
        ]

        os.makedirs(f"{self.base_path}/chunks", exist_ok=True)

        # Step 5: Persist all indices and companion configuration mappings.
        with bar(desc="Saving", color="cyan") as pbar:
            self.save()
            pbar.update(1)

    def save(self) -> None:
        """Saves matrix components and structured metadata onto the disk."""
        self.bm25_docs.save(self.docs_index_path)
        self.bm25_code.save(self.code_index_path)

        with open(self.docs_metadata_path, "w") as fd:
            json.dump(self.metadata_docs, fd, indent=4)

        with open(self.code_metadata_path, "w") as fd:
            json.dump(self.metadata_code, fd, indent=4)

    def load(self) -> None:
        """Loads index models and associated schema definitions from files."""
        # mmap=True uses memory-mapping to instantly read large indices
        # without consuming excess RAM overhead.
        self.bm25_docs = bm25s.BM25.load(self.docs_index_path, mmap=True)
        self.bm25_code = bm25s.BM25.load(self.code_index_path, mmap=True)

        if os.path.exists(self.docs_metadata_path):
            with open(self.docs_metadata_path, "r") as fd:
                self.metadata_docs = json.load(fd)

        if os.path.exists(self.code_metadata_path):
            with open(self.code_metadata_path, "r") as fd:
                self.metadata_code = json.load(fd)
