from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from student.models import MinimalSource
import re


def clean_source_content(content: str) -> str:
    # Return empty string if content is missing or only whitespace
    if not content or not content.strip():
        return ""

    # Split content into individual lines
    lines = content.splitlines()
    cleaned = []

    # Patterns for lines that should be discarded entirely
    DISCARD_PATTERNS = [
        re.compile(r"^\s*```"),                     # Code block delimiters
        re.compile(r"^\s*\?\?\?"),                  # MkDocs-style admonitions
        re.compile(r"^\s*#\s*--8<--"),              # Include markers
        re.compile(r"^\s*#\s*SPDX-"),               # SPDX license headers
        re.compile(r"^\s*\[]\(\)\{"),               # Empty link patterns
        re.compile(r"^\s*<(style|figure|antml:document)"),  # HTML tags to remove
        # re.compile(r"^\s*th\s*\{"),                 # CSS table header rules
        re.compile(r"^\s*(white-space|min-width)\s*:"),  # CSS properties
        re.compile(r"^\s*\}"),                      # Closing CSS braces
        re.compile(r"^={3,}$"),                     # Markdown separators
        re.compile(r"^\s*\|[\s\-|]*\|\s*$"),        # Empty Markdown table rows
    ]

    # Remove lines matching discard patterns
    for line in lines:
        if any(pat.match(line) for pat in DISCARD_PATTERNS):
            continue
        cleaned.append(line)

    # Rejoin cleaned lines
    text = "\n".join(cleaned)

    # Remove fenced code block markers like ```python
    text = re.sub(r"`{3}[\w]*\n?", "", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove admonition headers like !!! note
    text = re.sub(r"^\s*!!!\s+\w+\s*\n?", "", text, flags=re.MULTILINE)

    # Remove GitHub directive tags
    text = re.sub(r"<gh-(file|dir|issue|pr):[^>]+>", "", text)

    # Remove empty link placeholders
    text = re.sub(r"\[\]\(\)\{[^}]*\}", "", text)

    # Return cleaned text
    return text.strip()


def clean_search_results(search_results: list[dict]) -> list[dict]:
    # Clean the content of each retrieved source inside search results
    for result in search_results:
        for source in result.get("retrieved_sources", []):
            source["content"] = clean_source_content(source["content"])
    return search_results


class Chunker:
    def __init__(self, max_chunk_size: int) -> None:
        # Store maximum chunk size for splitting
        self.max_chunk_size = max_chunk_size

    def chunk_file(self, file: dict[str, str]) -> dict[str, dict]:
        # Extract path and content from the file dictionary
        path, content = next(iter(file.items()))
        # Split based on file extension
        return self.split(content, path.split('.')[-1].lower(), path)

    def chunk_text(self, text: str) -> list[str]:
        # Return empty list if text is empty or whitespace
        if not text or not text.strip():
            return []

        # Generic text splitter with fallback separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.2),
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        # Split and return cleaned chunks
        return [
            chunk.strip() for chunk in splitter.split_text(text)
            if chunk.strip()
        ]

    def split(self, content: str, f_type: str, path: str) -> dict[str, dict]:
        # Choose splitting strategy based on file type
        if f_type == 'py':
            # Use Python-aware splitting
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.max_chunk_size // 10,
            )
        elif f_type in ('md', 'rst'):
            # Use Markdown-aware splitting
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_chunk_size,
                chunk_overlap=int(self.max_chunk_size * 0.2),
            )
        else:
            # Generic fallback splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=int(self.max_chunk_size * 0.2),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )

        res = {}
        pos = 0

        # Iterate through generated chunks
        for i, chunk in enumerate(splitter.split_text(content)):
            # Find the chunk's position in the original content
            start = content.find(chunk, pos)
            if start == -1:
                start = pos
            end = start + len(chunk)
            pos = end

            # Clean the chunk content
            normalized = clean_source_content(chunk)

            # Skip chunks without meaningful alphanumeric content
            if not normalized or not re.search(r'[a-zA-Z0-9]', normalized):
                continue

            # Construct unique chunk identifier
            chunk_id = f"{path}:{start}:{end}:{i}"

            # Store chunk text and metadata
            res[chunk_id] = {
                "text": normalized,
                "source": MinimalSource(
                    file_path=path,
                    first_character_index=start,
                    last_character_index=end,
                    content=normalized,
                ),
            }

        return res
