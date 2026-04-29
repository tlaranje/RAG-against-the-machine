from .python_chunker import PythonChunker
from .text_chunker import TextChunker
from typing import List


class Chunker:
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size
        self.strategies = {
            'py': PythonChunker(),
            'md': TextChunker(),
            'txt': TextChunker()
        }

    def chunk_file(self, file_path: str, content: str) -> List[str]:
        extension = file_path.split('.')[-1].lower()

        strategy = self.strategies.get(extension, self.strategies['txt'])

        if not strategy:
            raise ValueError(
                f"No chunking strategy found for extension: {extension}"
            )

        return strategy.split(content, self.max_chunk_size)
