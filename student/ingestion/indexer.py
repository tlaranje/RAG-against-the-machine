from student.models import MinimalSource as MSource
from student.ingestion import Chunker, Parser
import bm25s
import json
import os


class Indexer:
    def __init__(
        self, parser: "Parser", base_path: str = "data/processed"
    ) -> None:
        self.parser = parser
        self.base_path = base_path
        self.bm25_path = f"{base_path}/bm25_index"
        self.metadata_path = f"{base_path}/chunks/metadata.json"
        self.bm25: bm25s.BM25 = bm25s.BM25()
        self.metadata: list["MSource"] = []

    def index(self, max_chunk_size: int) -> None:
        files = self.parser.parse_directory("data/raw/vllm-0.10.1")
        chunker = Chunker(max_chunk_size=max_chunk_size)
        chunks = {}
        for file_path, content in files.items():
            chunks.update(chunker.chunk_file({file_path: content}))
        chunk_tokens = bm25s.tokenize(list(chunks.keys()))
        self.bm25.index(chunk_tokens)
        self.metadata.extend(chunks.values())
        self.save()

    def save(self) -> None:
        self.bm25.save(self.bm25_path)
        os.makedirs(self.metadata_path.rsplit('/', 1)[0], exist_ok=True)
        with open(self.metadata_path, "w") as fd:
            json.dump([m.model_dump() for m in self.metadata], fd, indent=4)

    def load(self) -> None:
        self.bm25 = bm25s.BM25.load(self.bm25_path, mmap=True)

        with open(self.metadata_path, "r") as fd:
            data = json.load(fd)

        self.metadata = [MSource.model_validate(d) for d in data]
