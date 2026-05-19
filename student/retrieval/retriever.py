from typing import TYPE_CHECKING
from student.utils import bar
import bm25s
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from student.models import (
    MinimalSearchResults, RagDataset, StudentSearchResults,
    UnansweredQuestion, MinimalSource
)

if TYPE_CHECKING:
    from student.ingestion import Indexer


# ---------------------------------------------------------------------------
# Query expansion helpers
# ---------------------------------------------------------------------------

def _split_camel(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return text

def _split_dots(text: str) -> str:
    return re.sub(r"[./]", " ", text)

def _extract_keywords(text: str, min_len: int = 3) -> str:
    tokens = re.split(r"[\s,;:!?()\[\]{}'\"<>@#$%^&*+=|\\/-]+", text)
    return " ".join(t for t in tokens if len(t) >= min_len)

def _simple_stem(text: str) -> str:
    suffixes = [
        "ation", "ations", "ing", "ings", "ed", "er", "ers",
        "tion", "tions", "ment", "ments", "ness", "ity", "ies", "es", "s",
    ]
    stemmed = []
    for w in text.split():
        lw = w.lower()
        for suf in suffixes:
            if lw.endswith(suf) and len(lw) - len(suf) >= 4:
                lw = lw[: -len(suf)]
                break
        stemmed.append(lw)
    return " ".join(stemmed)

def _query_to_filename_variants(query: str) -> list[str]:
    """
    Try to derive likely Python filenames from a natural-language query.

    Examples:
      "fused batched MoE layer" -> ["fused_batched_moe.py", "fused_batched_moe"]
      "LLM class constructor"   -> ["llm.py", "llm"]
      "triton_flash_attention"  -> ["triton_flash_attention.py"]
    """
    # 1. grab all word-like tokens (ignore stopwords that aren't file hints)
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
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", query)
    meaningful = [t.lower() for t in tokens if t.lower() not in SKIP and len(t) >= 3]

    if not meaningful:
        return []

    variants = []

    # snake_case join of all meaningful tokens → filename
    snake = "_".join(meaningful)
    variants.append(snake + ".py")
    variants.append(snake)

    # try shorter prefixes (first 2, 3, 4 tokens) — catches "llm.py" from "LLM class"
    for n in range(1, min(5, len(meaningful))):
        short = "_".join(meaningful[:n])
        if len(short) >= 3:
            variants.append(short + ".py")
            variants.append(short)

    # also try individual meaningful tokens as filenames
    for t in meaningful:
        if len(t) >= 4:
            variants.append(t + ".py")

    return list(dict.fromkeys(variants))  # deduplicate preserving order


def expand_query(query: str) -> list[str]:
    base = query.strip()
    camel = _split_camel(base)
    no_underscore = base.replace("_", " ")
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
    for hint in _query_to_filename_variants(base):
        variants.add(hint)
    return [v.strip() for v in variants if v.strip()]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _term_coverage_score(query: str, content: str) -> float:
    """Fraction of unique query terms (>=3 chars) present in content."""
    q_terms = {t.lower() for t in re.split(r"\W+", query) if len(t) >= 3}
    if not q_terms:
        return 0.0
    content_lower = content.lower()
    return sum(1 for t in q_terms if t in content_lower) / len(q_terms)


def _filename_match_score(query: str, file_path: str) -> float:
    """
    Strong signal: does the file name (stem) match key terms in the query?

    Scores 0‥1 based on how many meaningful query tokens appear in the
    file stem. Returns 1.0 for an exact/near-exact match (e.g. query
    mentions "triton_flash_attention" and file is triton_flash_attention.py).
    """
    stem = Path(file_path).stem.lower()          # e.g. "fused_batched_moe"
    stem_tokens = set(re.split(r"[_\-.]", stem)) # {"fused","batched","moe"}

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
    q_tokens = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9]*", query)
                if t.lower() not in SKIP and len(t) >= 3}

    if not q_tokens or not stem_tokens:
        return 0.0

    matched = q_tokens & stem_tokens
    # score = overlap / min(|q|,|stem|) so short stems can still score high
    return len(matched) / min(len(q_tokens), len(stem_tokens))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, indexer: "Indexer") -> None:
        self.indexer = indexer

    def _is_valid_file(self, file_path: str, is_code_index: bool) -> bool:
        ext = Path(file_path).suffix.lower()
        if is_code_index:
            return ext == ".py"
        return ext in {".md", ".txt"}

    def _tokenize_query(self, query: str, is_code_index: bool):
        """Use the same tokenizer the Indexer used — critical for recall."""
        if is_code_index:
            return self.indexer.tokenizer_code([query])
        else:
            return self.indexer.tokenizer_docs([query])

    def _search_index(self, query: str, k: int, is_code_index: bool) -> list[dict]:
        bm25 = self.indexer.bm25_code if is_code_index else self.indexer.bm25_docs
        metadata = self.indexer.metadata_code if is_code_index else self.indexer.metadata_docs

        tokenized = self._tokenize_query(query, is_code_index)
        fetch_k = min(k * 30, len(metadata))
        results, scores = bm25.retrieve(tokenized, k=fetch_k)

        valid_chunks = []
        for i, idx in enumerate(results[0]):
            chunk = metadata[int(idx)]
            file_path = chunk.get("source", {}).get("file_path", "")
            if self._is_valid_file(file_path, is_code_index):
                chunk_copy = chunk.copy()
                chunk_copy["_score"] = float(scores[0][i])
                valid_chunks.append(chunk_copy)
            if len(valid_chunks) >= k:
                break
        return valid_chunks

    def _merge_and_rerank(self, all_results: list[dict], query: str) -> list[dict]:
        """
        Combine BM25 scores across query variants, then re-rank with:
          - term coverage in content  (semantic signal)
          - filename match score      (strongest signal for code queries)
        """
        score_map: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, dict] = {}

        for res in all_results:
            src = res["source"]
            uid = f"{src['file_path']}_{src['first_character_index']}"
            score_map[uid] += res["_score"]
            if uid not in chunk_map:
                chunk_map[uid] = res

        merged = []
        for uid, chunk in chunk_map.items():
            c = chunk.copy()
            bm25_acc = score_map[uid]
            content   = chunk.get("source", {}).get("content", "")
            file_path = chunk.get("source", {}).get("file_path", "")

            coverage  = _term_coverage_score(query, content)
            fn_match  = _filename_match_score(query, file_path)

            # filename match is the strongest signal for code — weight heavily
            c["_score"] = (
                bm25_acc  * 0.4
                + coverage * 8.0
                + fn_match * 20.0   # dominant for "what does X.py do" style queries
            )
            merged.append(c)

        return sorted(merged, key=lambda x: x["_score"], reverse=True)

    def search(self, prompt: str, k: int = 1, index_type: str = "both") -> list[dict]:
        """
        index_type: "code" -> apenas .py
                    "docs" -> apenas .md / .txt
                    "both" -> ambos
        """
        variants = expand_query(prompt)
        all_results: list[dict] = []

        for v in variants:
            if index_type in ("code", "both"):
                all_results.extend(self._search_index(v, k, is_code_index=True))
            if index_type in ("docs", "both"):
                all_results.extend(self._search_index(v, k, is_code_index=False))

        return self._merge_and_rerank(all_results, prompt)[:k]

    def build_context(self, chunks: list[dict], max_chars: int = 4000) -> str:
        parts = []
        total = 0
        for chunk in chunks:
            content = chunk.get("source", {}).get("content", "").strip()
            if not content:
                continue
            if total + len(content) > max_chars:
                break
            parts.append(content)
            total += len(content)
        return "\n---\n".join(parts)

    def search_dataset(self, data_path: str, k: int, save_dir: str) -> None:
        with bar(desc="Loading index", color="yellow") as pbar:
            self.indexer.load()
            pbar.update(1)

        dataset_name = Path(data_path).stem.lower()
        if "code" in dataset_name:
            index_type = "code"
        elif "docs" in dataset_name:
            index_type = "docs"
        else:
            index_type = "both"

        with open(data_path, "r") as fd:
            raw_data = json.load(fd)
        rag = RagDataset.model_validate(raw_data)

        results: list[MinimalSearchResults] = []
        for q in bar(rag.rag_questions, desc="Searching"):
            chunks = self.search(q.question, k=k, index_type=index_type)
            retrieved_sources = [
                MinimalSource(
                    file_path=chunk["source"]["file_path"],
                    first_character_index=chunk["source"]["first_character_index"],
                    last_character_index=chunk["source"]["last_character_index"],
                    content=chunk["source"]["content"],
                )
                for chunk in chunks
            ]
            results.append(MinimalSearchResults(
                question_id=str(q.question_id),
                question_str=q.question,
                retrieved_sources=retrieved_sources,
            ))

        if save_dir.endswith("/") or os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, "dataset.json")
        else:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            file_path = save_dir

        with bar(desc="Saving", color="cyan") as pbar:
            s_res = StudentSearchResults(search_results=results, k=k)
            with open(file_path, "w") as fd:
                json.dump(s_res.model_dump(), fd, indent=4, ensure_ascii=False)
            pbar.update(1)
