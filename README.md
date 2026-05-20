*This project has been created as part of the 42 curriculum by \<tlaranje\>.*

# RAG against the machine

## Description

This project implements a **Retrieval-Augmented Generation (RAG)** system capable of answering questions about the [vLLM](https://github.com/vllm-project/vllm) codebase. The pipeline ingests the vLLM repository, builds a searchable knowledge base, retrieves the most relevant code snippets and documentation for any question, and generates grounded natural-language answers using a local LLM (`Qwen/Qwen3-0.6B`).

The system is evaluated using **recall@k** metrics, measuring how often the correct source files are found among the top-k retrieved results.


## System Architecture

```
vLLM repository
      │
      ▼
  [ Parser ]          Scans .py, .md, .txt files recursively
      │
      ▼
  [ Chunker ]         Splits files into chunks (max 2000 chars, configurable)
      │               – Python-aware splitter for .py files
      │               – Markdown-aware splitter for .md/.rst files
      │               – Generic recursive splitter for everything else
      ▼
  [ Indexer ]         Builds two separate BM25 indices:
      │               – bm25_docs  (Markdown/text files, with stopword removal + stemming)
      │               – bm25_code  (Python files, exact tokens preserved)
      │               Saves indices + chunk metadata to data/processed/
      ▼
  [ Retriever ]       Query expansion → multi-variant BM25 search → reranking
      │               Heuristic reranker: BM25 score + term coverage + filename match
      ▼
  [ Generator ]       Builds few-shot prompt → runs Qwen GGUF model via llama-cpp
      │               Cleans and normalises raw model output
      ▼
  [ Evaluator ]       Computes recall@k using 5% character-overlap threshold
```


## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) as project and package manager

### Installation

```bash
make install
# or directly:
uv sync
```

### Index the repository

Place the vLLM repository at `data/raw/` (unzip `vllm-0.10.1.zip`), then:

```bash
uv run python -m student index --max_chunk_size 2000
```

Output indices are saved under `data/processed/`.

### Search a single query

```bash
uv run python -m student search "How does vLLM handle KV cache?" --k 5
```

### Answer a single question

```bash
uv run python -m student answer "How to configure the OpenAI compatible server?" --k 5
```

### Search a full dataset

```bash
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/dataset_docs_public.json
```

### Generate answers for a dataset

```bash
uv run python -m student answer_dataset \
  --student_search_results_path data/output/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/answers_docs.json
```

### Evaluate retrieval quality

```bash
./data/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results \
                  --student_answer_path ./data/output/dataset_docs.json \
                --dataset_path ./data/datasets/AnsweredQuestions/dataset_docs_public.json \
                  --k 5 --max_context_length 2000

```

### Makefile targets

| Target        | Description                              |
|---------------|------------------------------------------|
| `make install`| Install project dependencies via `uv`   |
| `make run`    | Run the main CLI entry point             |
| `make debug`  | Run with Python's `pdb` debugger         |
| `make clean`  | Remove `__pycache__`, `.mypy_cache`, etc.|
| `make lint`   | Run `flake8` + `mypy` checks             |


## Chunking Strategy

Two dedicated chunking strategies are applied based on file type:

**Python files (`.py`)** use LangChain's `Language.PYTHON` splitter, which respects function and class boundaries. Overlap is set to 10% of `max_chunk_size` to preserve method context across chunk boundaries.

**Markdown and RST files (`.md`, `.rst`)** use LangChain's `Language.MARKDOWN` splitter, which respects heading and paragraph boundaries. Overlap is set to 20%.

**All other files** fall back to a generic `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]` and 20% overlap.

After splitting, every chunk is passed through `clean_source_content`, which strips CSS declarations, MkDocs/RST directives, empty table rows, bare HTML tags, markdown image syntax, and other formatting noise. Chunks containing no alphanumeric content are discarded.

The default maximum chunk size is **2000 characters**, configurable via `--max_chunk_size`.


## Retrieval Method

The system uses **BM25** (via the `bm25s` library) as its primary retrieval algorithm, with two separate indices:

- `bm25_docs`: tokenised with English stopword removal and Porter stemming (`PyStemmer`), optimised for natural-language documentation queries.
- `bm25_code`: tokenised without stopwords or stemming, preserving exact identifiers, function names, and module paths.

### Query Expansion

Each query is transformed into multiple variants before searching:

- Original query, lowercased, and with underscores/hyphens replaced by spaces
- CamelCase splitting (`PagedAttentionScheduler` → `Paged Attention Scheduler`)
- Dot/slash splitting for module paths
- Keyword extraction (tokens ≥ 3 chars)
- Lightweight suffix stemming
- Filename variant generation (predicts likely `.py` module names from query terms)

All variants are searched against both indices. Results are merged by source UID.

### Reranking

After aggregating BM25 scores across variants, a heuristic reranker computes a final score:

```
final_score = bm25_accumulated * 0.4
            + term_coverage    * 8.0
            + filename_match   * 20.0
```

The heavy weight on `filename_match` is intentional: for code questions, getting the right file is far more important than textual similarity alone.


## Performance Analysis

Results on the public evaluation datasets with `k=5`, `--max_context_length=2000`:

| Metric     | Docs  | Code  |
|------------|-------|-------|
| Recall@1   | ~0.58 | ~0.29 |
| Recall@3   | ~0.85 | ~0.46 |
| Recall@5   | ~0.87 | ~0.51 |
| Recall@10  | ~0.87 | ~0.51 |

**Performance requirements met:**
- Indexing time: under 5 minutes
- Cold start latency: under 60 seconds
- Warm retrieval: 1000 questions under 90 seconds
- Recall@5: ≥ 80% on docs, ≥ 50% on code


## Design Decisions

**Two separate BM25 indices** rather than one unified index. Documentation and source code have fundamentally different token distributions. A single index would force a compromise on tokenisation strategy; splitting them lets each index be tuned independently.

**No neural embeddings in the mandatory part.** The `bm25s` library is extremely fast and CPU-friendly. Reaching the recall targets with BM25 + aggressive query expansion leaves room for semantic embeddings as a bonus without making the baseline brittle.

**`llama-cpp-python` for inference.** Quantised GGUF models run on CPU without a GPU, keeping the system portable. `cache_prompt=True` reuses the shared few-shot prefix across calls, cutting latency significantly on warm runs.

**Few-shot prompt with `/No_think` directive.** A single in-context example steers the model toward concise, source-grounded answers and suppresses chain-of-thought reasoning (which would exceed the 2-second-per-question budget on CPU).


## Challenges Faced

**Recall on code questions.** BM25 struggles when the question uses natural language but the answer lives in a Python file with terse identifiers. The `filename_match` heuristic and CamelCase query splitting provided the largest gains here.

**Chunk boundary alignment.** The evaluator requires ≥ 5% character overlap between retrieved and ground-truth spans. Chunking at heading boundaries rather than fixed character counts significantly reduced false negatives near section edges.

**LLM latency on CPU.** The 2-second-per-question limit is tight for a 0.6B model on CPU. Reducing `max_tokens` to 150, using greedy decoding (`temperature=0, top_k=1`), and sorting questions by length to improve KV-cache reuse collectively brought median latency well under the limit.

**Noisy documentation chunks.** MkDocs `--8<--` snippet-include directives, CSS blocks, and empty admonition headers inflated the index with zero-information chunks. The `clean_source_content` pipeline was built iteratively by inspecting the worst-scoring chunks manually.


## Example Usage

```bash
# 1. Index the repository
uv run python -m student index --max_chunk_size 2000

# 2. Ask a single question
uv run python -m student answer "What is the role of the KV cache manager?" --k 10

# 3. Batch search docs dataset
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/dataset_docs.json

# 4. Generate answers from search results
uv run python -m student answer_dataset \
  --student_search_results_path data/output/dataset_docs.json \
  --save_directory data/output/answers_docs.json
```


## Resources

### Documentation & Papers

- [BM25S library](https://github.com/xhluca/bm25s) — fast BM25 implementation used for indexing
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — language-aware chunking strategies
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Python bindings for GGUF model inference
- [Python Fire](https://github.com/google/python-fire) — CLI generation
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — base LLM used for generation
- Lewis et al. (2020) — [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401)

### AI Usage

AI was used for the following tasks in this project:

- **Regex patterns**: drafting and testing the `_DISCARD_LINE_PATTERNS` and `_INLINE_SUBS` lists in `chunker.py`.
- **Docstring writing**: generating Google-style docstrings for all public functions and classes.
- **Prompt engineering**: iterating on the few-shot prompt template and `clean_answer` post-processing pipeline.