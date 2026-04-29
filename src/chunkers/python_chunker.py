from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


class PythonChunker:
    def split(
        self, content: str, max_size: int
    ) -> dict[str, tuple[int, int]]:
        res = {}

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_size,
            chunk_overlap=max_size // 10
        )

        chunks = splitter.split_text(content)

        pos = 0
        for chunk in chunks:
            start = content.find(chunk, pos)
            end = start + len(chunk)
            pos = start
            res[chunk] = (start, end)

        return res
