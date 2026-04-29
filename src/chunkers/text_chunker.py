from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    def split(self, content: str, max_size: int) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size,
            chunk_overlap=int(max_size * 0.1),
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        return splitter.split_text(content)
