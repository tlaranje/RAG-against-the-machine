import json
import os
from typing import TYPE_CHECKING, List, Dict, Any

import bm25s
from student.ingestion import Chunker
from student.utils import bar

if TYPE_CHECKING:
    from student.ingestion import Parser


class Indexer:
    def __init__(
        self,
        parser: "Parser",
        base_path: str = "data/processed"
    ) -> None:
        self.parser = parser
        self.base_path = base_path
        self.bm25_path = f"{base_path}/bm25_index"
        self.metadata_path = f"{base_path}/chunks/metadata.json"
        self.bm25 = bm25s.BM25()
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_texts: List[str] = []

    def index(self, max_chunk_size: int) -> None:
        files = self.parser.parse_directory("data/raw/vllm-0.10.1")
        chunker = Chunker(max_chunk_size=max_chunk_size)
        chunks_data = {}

        with bar(total=len(files), desc="Chunking", color="green") as pbar:
            for path, content in files.items():
                chunks_data.update(chunker.chunk_file({path: content}))
                pbar.update(1)

        texts = [v["text"] for v in chunks_data.values()]
        sources = [v["source"] for v in chunks_data.values()]
        self.chunk_texts = texts

        with bar(desc="Tokenizing", color="blue") as pbar:
            tokens = bm25s.tokenize(texts)
            pbar.update(1)

        with bar(desc="Building Index", color="yellow") as pbar:
            self.bm25.index(tokens)
            pbar.update(1)

        self.metadata = []
        for meta, text in zip(sources, self.chunk_texts):
            self.metadata.append({
                "source": meta.model_dump(),
                "content": text
            })

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with bar(desc="Saving", color="cyan") as pbar:
            self.save()
            pbar.update(1)

    def save(self) -> None:
        self.bm25.save(self.bm25_path)

        with open(self.metadata_path, "w") as fd:
            json.dump(self.metadata, fd, indent=4)

    def load(self) -> None:
        self.bm25 = bm25s.BM25.load(self.bm25_path, mmap=True)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as fd:
                self.metadata = json.load(fd)
