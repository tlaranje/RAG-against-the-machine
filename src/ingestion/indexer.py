from src.models import MinimalSource
import bm25s
import json


class Indexer:
    def __init__(self, base_path: str = "data/processed") -> None:
        self.base_path = base_path
        self.bm25_path = f"{base_path}/bm25_index"
        self.metadata_path = f"{base_path}/chunks/metadata.json"
        self.bm25: bm25s.BM25 = bm25s.BM25()
        self.metadata: list["MinimalSource"] = []

    def index(self, chunks: dict[str, "MinimalSource"]) -> None:
        chunk_tokens = bm25s.tokenize(list(chunks.keys()))
        self.bm25.index(chunk_tokens)
        self.metadata.extend(chunks.values())

    def save(self) -> None:
        self.bm25.save(self.bm25_path)

        with open(self.metadata_path, "w") as fd:
            json.dump([m.model_dump() for m in self.metadata], fd, indent=4)

    def load(self) -> None:
        self.bm25 = bm25s.BM25.load(self.bm25_path)

        with open(self.metadata_path, "r") as fd:
            data = json.load(fd)

        self.metadata = [MinimalSource.model_validate(d) for d in data]
