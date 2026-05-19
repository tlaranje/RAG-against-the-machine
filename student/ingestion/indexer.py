from typing import TYPE_CHECKING, List, Dict, Any
from student.ingestion import Chunker
from student.utils import bar
from Stemmer import Stemmer
import bm25s
import json
import os

if TYPE_CHECKING:
    from student.ingestion import Parser


class Indexer:
    def __init__(
        self, parser: "Parser", base_path: str = "data/processed"
    ) -> None:
        """
        Initialise the Indexer with a parser and storage paths.

        Args:
            parser: Parser instance used to read raw documents.
            base_path: Root directory for storing the index and metadata.
        """
        self.parser = parser
        self.base_path = base_path
        self.bm25_path = f"{base_path}/bm25_index"
        self.metadata_path = f"{base_path}/chunks/metadata.json"
        self.bm25 = bm25s.BM25()
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_texts: List[str] = []
        self.stemmer = Stemmer("english")

    def index(self, max_chunk_size: int) -> None:
        """
        Parse, chunk, tokenize, and index all raw documents.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
        """
        files = self.parser.parse_directory("data/raw/vllm-0.10.1")
        chunker = Chunker(max_chunk_size=max_chunk_size)

        # Chunk every parsed file and merge results into a single dict.
        chunks_data = {}
        with bar(total=len(files), desc="Chunking", color="green") as pbar:
            for path, content in files.items():
                chunks_data.update(chunker.chunk_file({path: content}))
                pbar.update(1)

        # Separate texts and source metadata from the chunks dict.
        texts = [v["text"] for v in chunks_data.values()]
        sources = [v["source"] for v in chunks_data.values()]
        self.chunk_texts = texts

        # Tokenize with English stopword removal and stemming.
        with bar(desc="Tokenizing", color="blue") as pbar:
            tokens = bm25s.tokenize(
                texts, stopwords="en", stemmer=self.stemmer.stemWords
            )
            pbar.update(1)

        # Build the BM25 index from the tokenized corpus.
        with bar(desc="Building Index", color="yellow") as pbar:
            self.bm25.index(tokens)
            pbar.update(1)

        # Serialize source metadata to a JSON-compatible structure.
        self.metadata = []
        for meta in sources:
            self.metadata.append({"source": meta.model_dump()})

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Persist the index and metadata to disk.
        with bar(desc="Saving", color="cyan") as pbar:
            self.save()
            pbar.update(1)

    def save(self) -> None:
        """Save the BM25 index and chunk metadata to disk."""
        self.bm25.save(self.bm25_path)
        with open(self.metadata_path, "w") as fd:
            json.dump(self.metadata, fd, indent=4)

    def load(self) -> None:
        """Load the BM25 index and chunk metadata from disk."""
        self.bm25 = bm25s.BM25.load(self.bm25_path, mmap=True)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as fd:
                self.metadata = json.load(fd)
