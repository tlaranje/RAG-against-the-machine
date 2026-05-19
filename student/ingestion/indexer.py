from typing import TYPE_CHECKING, List, Dict, Any
from student.ingestion import Chunker
from student.utils import bar
from Stemmer import Stemmer
import bm25s
import json
import os

if TYPE_CHECKING:
    from student.ingestion import Parser


def is_code_file(path: str) -> bool:
    p = path.lower().strip()
    p = p.split(":")[0]
    p = p.split("#")[0]
    p = p.split("?")[0]
    return ".py" in p


class Indexer:
    def __init__(
        self, parser: "Parser", base_path: str = "data/processed"
    ) -> None:
        self.parser = parser
        self.base_path = base_path

        self.bm25_docs = bm25s.BM25()
        self.bm25_code = bm25s.BM25()

        self.metadata_docs: List[Dict[str, Any]] = []
        self.metadata_code: List[Dict[str, Any]] = []

        self.stemmer = Stemmer("english")

        # Exposed tokenizers so the Retriever can use the same ones
        self.tokenizer_docs = lambda texts: bm25s.tokenize(
            texts, stopwords="en", stemmer=self.stemmer.stemWords
        )
        self.tokenizer_code = lambda texts: bm25s.tokenize(
            texts, stopwords=[], stemmer=lambda x: x
        )

        self.docs_index_path = f"{base_path}/bm25_index_docs"
        self.code_index_path = f"{base_path}/bm25_index_code"
        self.docs_metadata_path = f"{base_path}/chunks/metadata_docs.json"
        self.code_metadata_path = f"{base_path}/chunks/metadata_code.json"

    def index(self, max_chunk_size: int = 1500) -> None:
        """
        Index all files under data/raw/vllm-0.10.1.
        max_chunk_size=1500 gives chunks large enough that BM25 has enough
        tokens to match on, while staying small enough to be precise.
        """
        files = self.parser.parse_directory("data/raw/vllm-0.10.1")
        chunker = Chunker(max_chunk_size=max_chunk_size)

        docs_texts, docs_sources = [], []
        code_texts, code_sources = [], []

        with bar(total=len(files), desc="Chunking", color="green") as pbar:
            for path, content in files.items():
                chunks = chunker.chunk_file({path: content})

                for _, data in chunks.items():
                    text = data["text"]
                    source = data["source"]
                    file_path = source.file_path

                    if is_code_file(file_path):
                        # Also prepend the file path as text so BM25 can match
                        # queries that mention a module/class name
                        enriched = f"{file_path}\n{text}"
                        code_texts.append(enriched)
                        code_sources.append(source)
                    else:
                        docs_texts.append(text)
                        docs_sources.append(source)

                pbar.update(1)

        with bar(desc="Tokenizing", color="blue") as pbar:
            docs_tokens = self.tokenizer_docs(docs_texts)
            code_tokens = self.tokenizer_code(code_texts)
            pbar.update(1)

        with bar(desc="Building Indexes", color="yellow") as pbar:
            self.bm25_docs.index(docs_tokens)
            self.bm25_code.index(code_tokens)
            pbar.update(1)

        self.metadata_docs = [{"source": s.model_dump()} for s in docs_sources]
        self.metadata_code = [{"source": s.model_dump()} for s in code_sources]

        os.makedirs(f"{self.base_path}/chunks", exist_ok=True)

        with bar(desc="Saving", color="cyan") as pbar:
            self.save()
            pbar.update(1)

    def save(self) -> None:
        self.bm25_docs.save(self.docs_index_path)
        self.bm25_code.save(self.code_index_path)

        with open(self.docs_metadata_path, "w") as fd:
            json.dump(self.metadata_docs, fd, indent=4)

        with open(self.code_metadata_path, "w") as fd:
            json.dump(self.metadata_code, fd, indent=4)

    def load(self) -> None:
        self.bm25_docs = bm25s.BM25.load(self.docs_index_path, mmap=True)
        self.bm25_code = bm25s.BM25.load(self.code_index_path, mmap=True)

        if os.path.exists(self.docs_metadata_path):
            with open(self.docs_metadata_path, "r") as fd:
                self.metadata_docs = json.load(fd)

        if os.path.exists(self.code_metadata_path):
            with open(self.code_metadata_path, "r") as fd:
                self.metadata_code = json.load(fd)
