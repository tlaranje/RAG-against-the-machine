from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from student.models import MinimalSource


class Chunker:
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk_file(self, file: dict[str, str]) -> dict[str, dict]:
        path, content = next(iter(file.items()))
        return self.split(content, path.split('.')[-1].lower(), path)

    def split(self, content: str, f_type: str, path: str) -> dict[str, dict]:
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

        res, pos = {}, 0
        for i, chunk in enumerate(splitter.split_text(content)):
            start = content.find(chunk, pos)
            if start == -1:
                continue

            end = start + len(chunk)
            pos = end
            chunk_id = f"{path}:{start}:{end}:{i}"

            res[chunk_id] = {
                "text": chunk,
                "source": MinimalSource(
                    file_path=path,
                    first_character_index=start,
                    last_character_index=end
                )
            }
        return res
