from typing import TYPE_CHECKING, Any
from collections import defaultdict
from student.utils import bar
from pathlib import Path
import json
import os
import re

from student.models import (
    MinimalSearchResults, MinimalSource,
    RagDataset, StudentSearchResults
)

if TYPE_CHECKING:
    from student.ingestion import Indexer

# Query Expansion Helpers


def _split_camel(text: str) -> str:
    """
    Splits CamelCase or camelCase strings into space-separated words.

    Args:
        text: The input string to split.

    Returns:
        A string with spaces inserted between camel case transitions.
    """
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return text


def _split_dots(text: str) -> str:
    """
    Replaces periods and forward slashes with spaces.

    Args:
        text: The input string containing dot notation or paths.

    Returns:
        The modified string with spaces instead of dots/slashes.
    """
    return re.sub(r"[./]", " ", text)


def _extract_keywords(text: str, min_len: int = 3) -> str:
    """
    Extracts alphanumeric tokens that meet a minimum length threshold.

    Args:
        text: The input text to parse.
        min_len: The minimum character length for a token to be kept.

    Returns:
        A space-separated string of the extracted valid keywords.
    """
    tokens = re.split(r"[\s,;:!?()\[\]{}'\"<>@#$%^&*+=|\\/-]+", text)
    return " ".join(t for t in tokens if len(t) >= min_len)


def _simple_stem(text: str) -> str:
    """
    Applies basic suffix stripping to approximate word stemming.

    Args:
        text: The space-separated words to stem.

    Returns:
        A lowercase string with common suffixes removed from its words.
    """
    # Common English suffixes to cut words down to their root form.
    suffixes = [
        "ation", "ations", "ing", "ings", "ed", "er", "ers",
        "tion", "tions", "ment", "ments", "ness", "ity", "ies", "es", "s",
    ]
    stemmed = []
    for w in text.split():
        lw = w.lower()
        for suf in suffixes:
            # Only chop off the suffix if the remaining base word is at least
            # 4 characters long (prevents over-truncating short words).
            if lw.endswith(suf) and len(lw) - len(suf) >= 4:
                lw = lw[:-len(suf)]
                break
        stemmed.append(lw)
    return " ".join(stemmed)


def _query_to_filename_variants(query: str) -> list[str]:
    """
    Derives potential Python filename patterns from a natural language query.

    Args:
        query: The user's natural language search query.

    Returns:
        A list of predicted filenames or module names derived from the query.
    """
    # Stopwords specific to programming syntax that we ignore when guessing
    # filenames (e.g., asking for a 'method' or 'class' doesn't mean the file
    # is named 'class.py').
    SKIP = {
        "what", "which", "where", "when", "does", "how", "the", "are",
        "for", "vllm", "default", "value", "values", "class", "method",
        "function", "module", "layer", "model", "returns", "return",
        "used", "uses", "with", "that", "this", "from", "into", "its",
        "constructor", "call", "calls", "struct", "structure", "array",
        "parameter", "parameters", "field", "fields", "type", "types",
        "shape", "constant", "constants", "format", "formats", "instead",
        "conditions", "must", "met", "about", "assert", "asserts",
        "compute", "computation",
    }
    # Matches words starting with a letter, followed by alphanumeric chars.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", query)
    meaningful = [
        t.lower() for t in tokens if t.lower() not in SKIP and len(t) >= 3
    ]

    if not meaningful:
        return []

    variants = []

    # Strategy 1: Combine all key terms using snake_case.
    snake = "_".join(meaningful)
    variants.append(snake + ".py")
    variants.append(snake)

    # Strategy 2: Create smaller partial sub-prefixes. If the user searches for
    # 'Triton flash attention layer', the file might just be 'triton_flash.py'.
    for n in range(1, min(5, len(meaningful))):
        short = "_".join(meaningful[:n])
        if len(short) >= 3:
            variants.append(short + ".py")
            variants.append(short)

    # Strategy 3: Try individual keywords as single files.
    for t in meaningful:
        if len(t) >= 4:
            variants.append(t + ".py")

    return list(dict.fromkeys(variants))


def _expand_query(query: str) -> list[str]:
    """
    Generates a collection of text variations to boost search recall.

    Args:
        query: The raw string query input.

    Returns:
        A list of clean, unique alternative representations of the query.
    """
    base = query.strip()
    camel = _split_camel(base)
    no_underscore = base.replace("_", " ")

    # We use a set structure here to automatically merge any duplicate
    # string transformations created by the helper functions.
    variants = {
        base,
        base.lower(),
        no_underscore,
        base.replace("-", " "),
        camel,
        _split_dots(base),
        _extract_keywords(base, min_len=3),
        _simple_stem(base),
        _simple_stem(camel),
        _simple_stem(no_underscore),
        _simple_stem(_extract_keywords(base, min_len=3)),
    }

    # Mix in our computed filename variations to improve source code matching.
    for hint in _query_to_filename_variants(base):
        variants.add(hint)

    # Clean out empty strings and return the final list.
    return [v.strip() for v in variants if v.strip()]


# Scoring Helpers

def _term_coverage_score(query: str, content: str) -> float:
    """
    Calculates the fraction of unique query terms found within a text block.

    Args:
        query: The base user query.
        content: The text payload of a retrieved document chunk.

    Returns:
        A ratio float between 0.0 and 1.0 indicating keyword coverage.
    """
    # Get all unique words from the query that are 3 characters or longer.
    q_terms = {t.lower() for t in re.split(r"\W+", query) if len(t) >= 3}
    if not q_terms:
        return 0.0
    content_lower = content.lower()
    # Count how many of those unique query terms appear inside the text chunk.
    return sum(1 for t in q_terms if t in content_lower) / len(q_terms)


def _filename_match_score(query: str, file_path: str) -> float:
    """
    Evaluates text match overlaps between a query and a file stem.

    Args:
        query: The raw search string.
        file_path: Relative or absolute path to the candidate file.

    Returns:
        An overlap score between 0.0 and 1.0 matching query terms to filenames.
    """
    # Path().stem gets only the filename without the directory or extension
    # (e.g., 'src/models/rag_pipeline.py' -> 'rag_pipeline').
    stem = Path(file_path).stem.lower()
    stem_tokens = set(re.split(r"[_\-.]", stem))

    SKIP = {
        "what", "which", "where", "when", "does", "how", "the", "are",
        "for", "vllm", "default", "value", "values", "class", "method",
        "function", "module", "layer", "model", "returns", "return",
        "used", "uses", "with", "that", "this", "from", "into", "its",
        "constructor", "call", "calls", "struct", "structure", "array",
        "parameter", "parameters", "field", "fields", "type", "types",
        "shape", "constant", "constants", "format", "formats", "instead",
        "conditions", "must", "met", "about", "assert", "asserts",
        "compute", "computation",
    }
    # Extract clean, lowercased query words that are not in the SKIP set.
    q_tokens = {
        t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9]*", query)
        if t.lower() not in SKIP and len(t) >= 3
    }

    if not q_tokens or not stem_tokens:
        return 0.0

    # Intersection set operation (&) finds terms present in BOTH collections.
    matched = q_tokens & stem_tokens
    # Normalized using min() to prevent penalizing very short filenames.
    return len(matched) / min(len(q_tokens), len(stem_tokens))


# Retriever Class

class Retriever:
    """Retrieves and ranks relevant snippets from documents or source code."""

    def __init__(self, indexer: "Indexer") -> None:
        """
        Initializes the retriever engine with a specific dataset indexer.

        Args:
            indexer: An Indexer instance managing access to BM25 indices.
        """
        self.indexer = indexer

    def _is_valid_file(self, file_path: str, is_code_index: bool) -> bool:
        """
        Validates if a file path fits the requested target index type.

        Args:
            file_path: The file path to check.
            is_code_index: True if searching source code, False if docs.

        Returns:
            True if the extension matches index parameters, otherwise False.
        """
        # Verifies the file suffix to make sure we don't accidentally load
        # non-code files into the code retriever pipeline or vice versa.
        ext = Path(file_path).suffix.lower()
        if is_code_index:
            return ext == ".py"
        return ext in {".md", ".txt"}

    def _tokenize_query(self, query: str, is_code_index: bool) -> Any:
        """
        Tokenizes a query utilizing the indexer's internal strategies.

        Args:
            query: The text snippet to tokenize.
            is_code_index: Flag picking either the code or docs pipeline.

        Returns:
            A list of token streams generated by the indexer configuration.
        """
        # Crucial for performance: we process the query using the exact same
        # logic used to build the BM25 index database.
        if is_code_index:
            return self.indexer.tokenizer_code([query])
        else:
            return self.indexer.tokenizer_docs([query])

    def _search_index(
        self, query: str, k: int, is_code_index: bool
    ) -> list[dict[str, Any]]:
        """
        Queries the underlying BM25 index for raw matches.

        Args:
            query: The expanded variant text string.
            k: The absolute maximum number of document hits to process.
            is_code_index: Targeted selection switch choosing code vs docs.

        Returns:
            A list of retrieved metadata dictionaries containing scores.
        """
        # Dynamically switch indices and tables based on the search mode.
        if is_code_index:
            bm25 = self.indexer.bm25_code
            metadata = self.indexer.metadata_code
        else:
            bm25 = self.indexer.bm25_docs
            metadata = self.indexer.metadata_docs

        tokenized = self._tokenize_query(query, is_code_index)
        # Oversampling (k * 30): We request a much larger batch than 'k' from
        # the engine because many results might get rejected by _is_valid_file.
        fetch_k = min(k * 30, len(metadata))
        results, scores = bm25.retrieve(tokenized, k=fetch_k)

        valid_chunks = []
        # results[0] contains the matched index IDs from the search matrix.
        for i, idx in enumerate(results[0]):
            chunk = metadata[int(idx)]
            file_path = chunk.get("source", {}).get("file_path", "")
            if self._is_valid_file(file_path, is_code_index):
                chunk_copy = chunk.copy()
                chunk_copy["_score"] = float(scores[0][i])
                valid_chunks.append(chunk_copy)
            # Terminate early once we have accumulated enough valid entries.
            if len(valid_chunks) >= k:
                break
        return valid_chunks

    def _merge_and_rerank(
        self, all_results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """
        Merges results across variations and applies score heuristics.

        Args:
            all_results: Collected match dictionaries from multiple runs.
            query: The unexpanded foundational prompt.

        Returns:
            A descending sorted collection of unified chunk outputs.
        """
        # defaultdict(float) initializes missing keys with 0.0 automatically.
        score_map: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, dict[str, Any]] = {}

        # Combine scores for identical text snippets matched across variants.
        for res in all_results:
            src = res["source"]
            # Unique ID formed by the path and its character location.
            uid = f"{src['file_path']}_{src['first_character_index']}"
            score_map[uid] += res["_score"]
            if uid not in chunk_map:
                chunk_map[uid] = res

        merged = []
        for uid, chunk in chunk_map.items():
            c = chunk.copy()
            bm25_acc = score_map[uid]
            content = chunk.get("source", {}).get("content", "")
            file_path = chunk.get("source", {}).get("file_path", "")

            coverage = _term_coverage_score(query, content)
            fn_match = _filename_match_score(query, file_path)

            # Heuristic reranker: BM25 provides the baseline, text coverage
            # adds a semantic verification, and matching filename tokens gets
            # an aggressive weight bonus (critical for code questions).
            c["_score"] = (
                bm25_acc * 0.4
                + coverage * 8.0
                + fn_match * 20.0
            )
            merged.append(c)

        # Sort the accumulated entries in descending order based on the score.
        return sorted(merged, key=lambda x: x["_score"], reverse=True)

    def search(
        self, prompt: str, k: int = 1, index_type: str = "both"
    ) -> list[dict[str, Any]]:
        """
        Executes full retrieval using query expansion options.

        Args:
            prompt: Original raw search assertion.
            k: Upper count bounding total requested records.
            index_type: Selection string choice: 'code', 'docs', or 'both'.

        Returns:
            Re-ranked list containing up to k result representations.
        """
        variants = _expand_query(prompt)
        all_results: list[dict[str, Any]] = []

        # Run individual search tasks across every single query variant.
        for v in variants:
            if index_type in ("code", "both"):
                all_results.extend(
                    self._search_index(v, k, is_code_index=True)
                )
            if index_type in ("docs", "both"):
                all_results.extend(
                    self._search_index(v, k, is_code_index=False)
                )

        # Merge matching references and apply weighting rules to pick top k.
        return self._merge_and_rerank(all_results, prompt)[:k]

    def build_context(
        self, chunks: list[dict[str, Any]], max_chars: int = 4000
    ) -> str:
        """
        Assembles valid content text segments safely up to character limit.

        Args:
            chunks: Document data elements containing target contents.
            max_chars: Character ceiling to protect window limits.

        Returns:
            A clean string containing textual blocks separated by rules.
        """
        parts = []
        total = 0
        for chunk in chunks:
            content = chunk.get("source", {}).get("content", "").strip()
            if not content:
                continue
            # Budget gatekeeper: stop accumulating blocks if we exceed limit.
            if total + len(content) > max_chars:
                break
            parts.append(content)
            total += len(content)
        return "\n---\n".join(parts)

    def search_dataset(self, data_path: str, k: int, save_dir: str) -> None:
        """
        Processes questions from a RAG JSON file and saves results.

        Args:
            data_path: Location of target input dataset JSON file.
            k: Match count limits passed onto underlying execution engine.
            save_dir: Location directory or exact file where results are save.
        """
        with bar(desc="Loading index", color="yellow") as pbar:
            self.indexer.load()
            pbar.update(1)

        # Choose the processing mode based on keywords in the file name.
        dataset_name = Path(data_path).stem.lower()
        if "code" in dataset_name:
            index_type = "code"
        elif "docs" in dataset_name:
            index_type = "docs"
        else:
            index_type = "both"

        with open(data_path, "r") as fd:
            raw_data = json.load(fd)
        # Parse and validate the raw dict payload against Pydantic definition.
        rag = RagDataset.model_validate(raw_data)

        results = []
        for q in bar(rag.rag_questions, desc="Searching"):
            chunks = self.search(q.question, k=k, index_type=index_type)
            retrieved_sources = [
                MinimalSource(
                    file_path=chunk["source"]["file_path"],
                    first_character_index=chunk["source"][
                        "first_character_index"
                    ],
                    last_character_index=chunk["source"][
                        "last_character_index"
                    ],
                    content=chunk["source"]["content"],
                )
                for chunk in chunks
            ]
            results.append(MinimalSearchResults(
                question_id=str(q.question_id),
                question_str=q.question,
                retrieved_sources=retrieved_sources,
            ))

        # Check if target is an explicit directory path or a target filename.
        if save_dir.endswith("/") or os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, "dataset.json")
        else:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            file_path = save_dir

        with bar(desc="Saving", color="cyan") as pbar:
            s_res = StudentSearchResults(search_results=results, k=k)
            with open(file_path, "w") as fd:
                json.dump(
                    s_res.model_dump(), fd, indent=4, ensure_ascii=False
                )
            pbar.update(1)
