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

        self.bm25 = bm25s.BM25()

        self.metadata_docs: list[dict[str, Any]] = []
        self.metadata_code: list[dict[str, Any]] = []

        self.stemmer = Stemmer("english")

        self.tokenizer: Any = lambda texts: bm25s.tokenize(
            texts, stopwords="en"
        )

        self.index_path = f"{base_path}/bm25_index"
        self.docs_metadata_path = f"{base_path}/chunks/metadata_docs.json"
        self.code_metadata_path = f"{base_path}/chunks/metadata_code.json"
        self.metadata_all_path = f"{base_path}/chunks/metadata_all.json"

        self.metadata_all: list[dict[str, Any]] = []

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
            all_texts = docs_texts + code_texts
            all_tokens: Any = self.tokenizer(all_texts)
            pbar.update(1)

        # Step 3: Fit the BM25 statistical matrix with the token datasets.
        with bar(desc="Building Index", color="yellow") as pbar:
            self.bm25.index(all_tokens)
            pbar.update(1)

        # Step 4: Serialize source positions into lightweight dictionaries.
        self.metadata_docs = [{"source": s.model_dump()} for s in docs_sources]
        self.metadata_code = [{"source": s.model_dump()} for s in code_sources]
        self.metadata_all = self.metadata_docs + self.metadata_code

        os.makedirs(f"{self.base_path}/chunks", exist_ok=True)

        # Step 5: Persist all indices and companion configuration mappings.
        with bar(desc="Saving", color="cyan") as pbar:
            self.save()
            pbar.update(1)

    def save(self) -> None:
        self.bm25.save(self.index_path)
        with open(self.docs_metadata_path, "w") as fd:
            json.dump(self.metadata_docs, fd, indent=4)
        with open(self.code_metadata_path, "w") as fd:
            json.dump(self.metadata_code, fd, indent=4)
        with open(self.metadata_all_path, "w") as fd:
            json.dump(self.metadata_all, fd, indent=4)

    def load(self) -> None:
        self.bm25 = bm25s.BM25.load(self.index_path, mmap=True)
        if os.path.exists(self.metadata_all_path):
            with open(self.metadata_all_path, "r") as fd:
                self.metadata_all = json.load(fd)
