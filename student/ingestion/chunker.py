from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from student.models import MinimalSource
import re
import os

_DISCARD_LINE_PATTERNS = [
    # code fence delimiters  ```python / ``` alone
    re.compile(r"^\s*```"),
    # MkDocs admonitions  ??? note
    re.compile(r"^\s*\?\?\?"),
    # include markers  # --8<--
    re.compile(r"^\s*#\s*--8<--"),
    # SPDX license headers
    re.compile(r"^\s*#\s*SPDX-"),
    # HTML structural tags that never carry prose
    re.compile(r"^\s*<(style|figure|antml:document|details|summary)"),
    # CSS property declarations
    re.compile(
        r"^\s*(white-space|min-width|font-size|color|margin|padding)\s*:"
    ),
    # lone closing braces / brackets left over from code blocks
    re.compile(r"^\s*[}\]]\s*$"),
    # long ======= separators
    re.compile(r"^={3,}$"),
    # empty or pure-separator Markdown table rows  | --- | --- |
    re.compile(r"^\s*\|[\s\-:|]*\|\s*$"),
    # horizontal rules  ---  or  ───
    re.compile(r"^\s*[-–]{3,}\s*$"),
    # lone ellipsis  ...
    re.compile(r"^\s*\.\.\.\s*$"),
    # table-of-contents markers  [TOC]
    re.compile(r"^\s*\[TOC\]\s*$", re.IGNORECASE),
    # single-line HTML comments  <!-- ... -->
    re.compile(r"^\s*<!--.*?-->\s*$"),
    # lone period
    re.compile(r"^\s*\.\s*$"),
    # empty blockquotes  >
    re.compile(r"^\s*>\s*$"),
    # lone double colon  ::
    re.compile(r"^\s*::\s*$"),
]

# Each entry is (compiled_pattern, replacement_string).
# Applied in order after line-level filtering.
_INLINE_SUBS = [
    # fenced code block markers  ```lang  or  ``` alone
    (re.compile(r"`{3}[\w]*\n?"), ""),
    # GitHub directive tags  <gh-file:path>, <gh-issue:123>, etc.
    (re.compile(r"<gh-(file|dir|issue|pr|project):[^>]+>"), ""),
    # empty link placeholders  [](){ ... }
    (re.compile(r"\[\]\(\)\{[^}]*\}"), ""),
    # admonition headers  !!! note "title"
    (re.compile(r"^\s*!!!\s+\w+.*\n?", re.MULTILINE), ""),
    # Sphinx / RST directives  .. code-block:: python
    (re.compile(r"^\s*\.\. \w[\w:-]*::.*\n?", re.MULTILINE), ""),
    # MkDocs include tags  --8<-- "path/to/file"
    (re.compile(r"--8<--\s+\"[^\"]+\""), ""),
    # inline HTML tags that carry no text  <br>, <nobr>, <span>, etc.
    (re.compile(
        r"</?(?:br|nobr|p|span|div|b|i|sup|sub|code)[^>]*>", re.IGNORECASE
    ), ""),
    # Markdown image syntax  ![alt text](url)  – no semantic value for Q&A
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
    # reference-style links  [visible text][kebab-ref]  → keep visible text
    (re.compile(r"\[([^\]]+)\]\[[a-z][a-z0-9-]*\]"), r"\1"),
    # trailing whitespace on every line
    (re.compile(r"[ \t]+$", re.MULTILINE), ""),
    # collapse three or more consecutive blank lines into two
    (re.compile(r"\n{3,}"), "\n\n"),
]


def clean_source_content(content: str) -> str:
    """
    Remove noise from raw documentation source text.

    Strips structural and formatting artefacts that carry no semantic value
    for question-answering (CSS declarations, bare HTML tags, empty table
    rows, image references, MkDocs/RST directives, …) while preserving
    prose, code examples, and inline markup that explains *what* or *how*.

    Args:
        content: Raw text extracted from a documentation source.

    Returns:
        Cleaned text with redundant blank lines collapsed, or an empty
        string when the input is blank or reduces to nothing after cleaning.
    """
    if not content or not content.strip():
        return ""

    # pass 1: line-level filtering
    cleaned_lines = []
    for line in content.splitlines():
        if any(pat.match(line) for pat in _DISCARD_LINE_PATTERNS):
            continue
        # Drop lines that contain no word characters at all (pure symbols)
        stripped = line.strip()
        if stripped and not re.search(r"\w", stripped):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # pass 2: inline / regex substitutions
    for pattern, replacement in _INLINE_SUBS:
        text = pattern.sub(replacement, text)

    return text.strip()


def clean_search_results(search_results: list[dict]) -> list[dict]:
    """
    Apply ``clean_source_content`` to every source in a results list.

    Mutates each source dict in-place so the caller always receives the
    same list object with cleaned ``content`` fields.

    Args:
        search_results: List of result dicts, each containing a
            ``retrieved_sources`` key whose value is a list of source
            dicts with at least a ``"content"`` field.

    Returns:
        The same list with all source ``content`` fields cleaned in-place.
    """
    for result in search_results:
        for source in result.get("retrieved_sources", []):
            source["content"] = clean_source_content(source["content"])
    return search_results


class Chunker:
    """
    Split documents and plain text into overlapping chunks.

    Selects a language-aware splitter for Python (``.py``) and Markdown /
    RST (``.md``, ``.rst``) files, and falls back to a generic recursive
    splitter for everything else.  Every chunk is cleaned with
    ``clean_source_content`` before being stored, and chunks that contain
    no alphanumeric characters are discarded.

    Attributes:
        max_chunk_size: Maximum number of characters per chunk.
    """

    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk_file(self, file: dict[str, str]) -> dict[str, dict]:
        """
        Chunk a single file given as a path-to-content mapping.

        Derives the file type from the extension and delegates to
        :meth:`split`.

        Args:
            file: Single-entry dict mapping one file path to its full text.

        Returns:
            Dict mapping chunk IDs to chunk text and metadata.
            See :meth:`split` for the exact structure.
        """
        path, content = next(iter(file.items()))
        f_type = os.path.splitext(path)[1].lower().lstrip(".")
        return self.split(content, f_type, path)

    def chunk_text(self, text: str) -> list[str]:
        """
        Split a plain text string into overlapping chunks.

        Uses a generic :class:`RecursiveCharacterTextSplitter` with an
        overlap of 20 % of ``max_chunk_size`` to preserve context across
        chunk boundaries.

        Args:
            text: Arbitrary plain text to split.

        Returns:
            List of non-empty, stripped chunk strings. Returns an empty
            list when ``text`` is blank or whitespace-only.
        """
        if not text or not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.2),
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]

    def split(self, content: str, f_type: str, path: str) -> dict[str, dict]:
        """
        Split file content into cleaned, position-annotated chunks.

        Chooses a splitter based on file type:

        * ``"py"`` → :data:`Language.PYTHON` with 10 % overlap.
        * ``"md"`` / ``"rst"`` → :data:`Language.MARKDOWN` with 20 % overlap.
        * anything else → generic recursive splitter with 20 % overlap.

        Each chunk is cleaned with :func:`clean_source_content`.  Chunks
        that are empty or contain no alphanumeric characters after cleaning
        are silently discarded.

        Args:
            content: Full text content of the file.
            f_type: Lowercase file extension (e.g. ``"py"``, ``"md"``).
            path: File path used to build unique chunk IDs and to populate
                :class:`~student.models.MinimalSource` metadata.

        Returns:
            Dict mapping ``"<path>:<start>:<end>:<index>"`` keys to::

                {
                    "text": str,
                    "source": MinimalSource,
                }

            Returns an empty dict when no valid chunks are produced.
        """
        if f_type == "py":
            # Python-aware splitter preserves function / class boundaries
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.max_chunk_size // 10,
            )
        elif f_type in ("md", "rst"):
            # Markdown-aware splitter respects heading boundaries
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_chunk_size,
                chunk_overlap=int(self.max_chunk_size * 0.2),
            )
        else:
            # Generic fallback for plain text and unknown file types
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=int(self.max_chunk_size * 0.2),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )

        res: dict[str, dict] = {}
        # Track search position to handle duplicate substrings correctly
        pos = 0

        for i, chunk in enumerate(splitter.split_text(content)):
            # Locate the chunk within the original content string so we
            # can record accurate character-level source positions.
            start = content.find(chunk, pos)
            if start == -1:
                # Fallback: find failed (e.g. chunk was modified); use pos
                start = pos
            end = start + len(chunk)
            pos = end

            normalized = clean_source_content(chunk)
            # Discard chunks that carry no alphanumeric information
            if not normalized or not re.search(r"[a-zA-Z0-9]", normalized):
                continue

            # Unique ID encodes path, byte range, and chunk index to allow
            # deterministic lookup and deduplication downstream.
            chunk_id = f"{path}:{start}:{end}:{i}"
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
