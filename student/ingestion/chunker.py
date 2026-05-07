from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from student.models import MinimalSource


class Chunker:
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk_file(self, file: dict[str, str]) -> dict[str, "MinimalSource"]:
        file_path, content = next(iter(file.items()))
        extension = file_path.split('.')[-1].lower()
        return self.split(content, extension, file_path)

    def split(
        self, content: str, f_type: str, file_path: str
    ) -> dict[str, "MinimalSource"]:
        res = {}

        if f_type == 'py':
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.max_chunk_size // 10
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=int(self.max_chunk_size * 0.1),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )

        chunks = splitter.split_text(content)
        pos = 0
        for chunk in chunks:
            start = content.find(chunk, pos)
            end = start + len(chunk)
            pos = start
            res[chunk] = MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            )

        return res
