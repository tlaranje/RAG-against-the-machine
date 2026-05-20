from contextlib import redirect_stderr
from student.ingestion import Chunker
from student.utils import bar
from llama_cpp import Llama
from typing import Any
from rich import print
import time
import json
import os
import re
from student.models import (
    MinimalAnswer, StudentSearchResults, StudentSearchResultsAndAnswer,
    MinimalSearchResults
)

# Maximum number of new tokens the model may produce per answer.
_MAX_NEW_TOKENS = 150

# Number of top-ranked chunks concatenated as context for each answer.
_NUM_CHUNKS = 2

# Maximum number of tokens in the model's context window (prompt + answer).
_N_CTX = 2048

# Maximum number of characters per text chunk produced by the Chunker.
_CHUNK_SIZE = 160

# Number of tokens evaluated in parallel during prompt processing.
_N_BATCH = 512

# Number of questions processed in a single forward-pass batch.
_BATCH_SIZE = 8

# Answers that take longer than this (seconds) are flagged as slow.
_TIMEOUT_SECONDS = 2.0

# Leave one logical CPU free to avoid starving the OS scheduler.
_N_THREADS = (os.cpu_count() or 4) - 1

# A single demonstration that conditions the model to:
#   • answer only from the supplied context
#   • skip chain-of-thought (/No_think directive for Qwen-R1 variants)
#   • avoid markdown, URLs, and preamble phrases
FEW_SHOT = (
    "/No_think "
    "Instructions: Answer using ONLY the context. "
    "Be direct. No preamble. "
    "No markdown. No URLs. "
    "If it is a list use commas. "
    "If not in context: 'Not found in context'.\n\n"
    "Context: The three key abstractions used for disaggregated prefilling "
    "in vLLM are: KV pipe, KV lookup buffer, and KV connector.\n"
    "Question: What are the three key abstractions used for disaggregated "
    "prefilling in vLLM?\n"
    "Answer: KV pipe, KV lookup buffer, and KV connector\n\n"
)

# Strip Qwen-R1 chain-of-thought blocks (inclusive of tags).
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Remove anything that looks like the start of a new prompt turn so
# the model cannot "hallucinate" extra Q&A pairs in its output.
_RE_NEW_TURN = re.compile(
    r"(\nAnswer:.*|\nQuestion:.*|\nContext:.*|\nInstructions:.*)$",
    re.DOTALL | re.IGNORECASE,
)

# Common preamble phrases that add no information value.
_RE_PREAMBLE = re.compile(
    r"^(Answer\s*:\s*|The answer is[:\s]+|Okay[,\s]+)",
    re.IGNORECASE,
)

# Fenced and inline code blocks — irrelevant for factoid answers.
_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```|`[^`\n]+`")

# Markdown hyperlinks: kept as "label (url)" for location questions,
# collapsed to just "label" otherwise.
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Bare URLs are suppressed unless the question asks "where".
_RE_URL = re.compile(r"https?://\S+")

# ATX-style markdown headers (e.g., "## Section Title").
_RE_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def clean_answer(text: str, question: str = "") -> str:
    """
    Normalise raw model output into a clean, plain-text answer.
    The function applies a sequence of regex substitutions to remove
    artefacts introduced by the prompt format or the model itself
    (chain-of-thought blocks, preamble phrases, markdown, URLs).
    Whitespace is collapsed and trailing punctuation is stripped.

    Args:
        text: Raw text produced by the language model.
        question: The original question string.  Used to decide
            whether URLs/links should be preserved (when the question
            starts with "where").

    Returns:
        A cleaned, single-line answer string. May be empty if the
        model produced no meaningful content.
    """
    # Remove <think>…</think> reasoning blocks first so subsequent
    # patterns operate on the final answer only.
    text = _RE_THINK.sub("", text)

    # Truncate at any spurious follow-up turn the model may have
    # generated beyond the first "Answer:" line.
    text = _RE_NEW_TURN.sub("", text)

    # Drop common boilerplate openers.
    text = _RE_PREAMBLE.sub("", text)

    # Inline/fenced code blocks are never useful in factoid answers.
    text = _RE_CODE_BLOCK.sub("", text)

    # Strip leading "#" header markers.
    text = _RE_MD_HEADER.sub("", text)

    text = _RE_MD_LINK.sub(r"\1 (\2)", text)

    # Remove trailing separators that can appear after list items.
    text = text.rstrip(" ,-")

    return " ".join(text.split()).strip()


def _print_slow_answers(slow_answers: list[dict[str, Any]]) -> None:
    """
    Print a sorted report of answers that exceeded the timeout.

    Args:
        slow_answers: List of dicts, each containing the keys
            ``question`` (str), ``answer`` (str), and ``time``
            (float, seconds elapsed).  An empty list produces a
            success message instead of a report.
    """
    if not slow_answers:
        print("[bold green]No answers took longer than 2s.[/bold green]")
        return

    slow_answers.sort(key=lambda x: x["time"], reverse=True)

    print(
        f"\n[bold red]{len(slow_answers)} answer(s) exceeded "
        f"{_TIMEOUT_SECONDS:.1f}s:[/bold red]"
    )
    for entry in slow_answers:
        print(
            f"\n[yellow]{entry['time']:.2f}s[/yellow]"
            f" | [cyan]{entry['question']}[/cyan]"
        )
        print(f"→ {entry['answer']}")


def _make_llama_instance(
    model_path: str, n_ctx: int, n_threads: int, n_batch: int,
) -> Llama:
    """
    Instantiate a llama-cpp ``Llama`` model with stderr suppressed.

    Llama.cpp emits verbose loading logs to stderr by default.
    Redirecting stderr to ``/dev/null`` keeps the terminal clean
    during application start-up.

    Args:
        model_path: Absolute or relative path to the GGUF model file.
        n_ctx: Context window length in tokens.
        n_threads: Number of CPU threads used for inference.
        n_batch: Prompt-evaluation batch size (tokens per chunk).

    Returns:
        A ready-to-use ``Llama`` instance with mlock enabled and
        prompt caching active.
    """
    with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            # Pin model weights in RAM to prevent OS swapping.
            use_mlock=True,
            verbose=False,
            # Reuse the KV cache across calls with a shared prefix.
            cache_prompt=True,
        )


def _quick_sort_similarity(
    all_results: list[MinimalSearchResults]
) -> list[Any]:
    """
    Sort search results to group similar-length questions together.
    Placing questions of similar token length consecutively reduces
    padding waste when the underlying runtime uses batched decoding.
    The primary key is question length, the secondary key is the
    question text itself for a stable, reproducible order.

    Args:
        all_results: Flat list of search-result objects, each
            expected to have a ``question`` attribute (str).

    Returns:
        A new sorted list; the original list is not mutated.
        Returns an empty list when the input is empty.
    """
    if not all_results:
        return []
    # Sort by (lexicographic, character length) for a cache-friendly
    # ordering without the overhead of a true similarity computation.
    return sorted(
        all_results, key=lambda x: (len(x.question_str), x.question_str)
    )


class SmallLLM:
    """
    CPU-friendly inference wrapper around a quantised Qwen GGUF model.

    Loads the model once at construction time and exposes a
    ``generate_batch`` method that runs prompts sequentially.

    Raises:
        FileNotFoundError: If the model file is absent at the path
            derived from the ``HF_HOME`` environment variable.
    """
    def __init__(self) -> None:
        # Resolve the model path from the environment, falling back to
        # a local ".llm" directory when HF_HOME is not set.
        hf_home = os.getenv("HF_HOME", "./.llm")
        model_path = os.path.join(hf_home, "Qwen")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self._model = _make_llama_instance(
            model_path, _N_CTX, _N_THREADS, _N_BATCH
        )

    def _run_batch(self, prompts: list[str]) -> list[str]:
        """
        Run inference on a list of prompt strings sequentially.

        Each non-empty prompt is passed to the model with greedy
        decoding (temperature=0, top_k=1). Stop tokens prevent the
        model from generating beyond the first answer turn.

        Args:
            prompts: Pre-formatted prompt strings. Empty strings are
                passed through as empty outputs without an LLM call.

        Returns:
            A list of raw answer strings parallel to ``prompts``.
        """
        outputs = []
        for prompt in prompts:
            # Skip blank prompts that arise from empty retrieval sets.
            if not prompt.strip():
                outputs.append("")
                continue

            res: Any = self._model(
                prompt=prompt,
                max_tokens=_MAX_NEW_TOKENS,
                # Greedy decoding — deterministic and fastest on CPU.
                temperature=0.0,
                top_k=1,
                # Stop at any token that would begin a new prompt turn.
                stop=["Question:", "Context:", "Instructions:", "\n\n"],
            )
            outputs.append(res["choices"][0]["text"].strip())
        return outputs

    def generate_batch(
        self, prompts: list[str]
    ) -> list[tuple[str, bool, float]]:
        """
        Generate answers for a batch of prompts with timing.

        Measures time across the whole batch and marks
        answers as timed-out when the per-question average exceeds
        ``_TIMEOUT_SECONDS``.  On any exception the batch falls back
        to empty strings so the caller can continue processing.

        Args:
            prompts: Pre-formatted prompt strings to run through the
                model.

        Returns:
            A list of ``(answer, timed_out, elapsed_seconds)`` tuples
            parallel to ``prompts``.  ``timed_out`` is ``True`` when
            the per-question average exceeded ``_TIMEOUT_SECONDS``.
        """
        results = []
        t0 = time.perf_counter()
        try:
            answers = self._run_batch(prompts)
            elapsed = (time.perf_counter() - t0) / max(len(prompts), 1)
            timed_out = elapsed > _TIMEOUT_SECONDS
            for ans in answers:
                results.append((ans, timed_out, elapsed))
        except Exception as exc:
            print(f"[red]Error processing batch: {exc}[/red]")
            # Return neutral sentinel values so the pipeline continues.
            for _ in prompts:
                results.append(("", False, 0.0))
        return results


class Generator:
    """
    End-to-end question-answering pipeline using a small local LLM.

    Combines a ``Chunker`` (for context splitting) with ``SmallLLM``
    (for answer generation) and provides two entry points:

    * ``answer`` — interactive single-question mode.
    * ``answer_dataset`` — batch mode that reads a JSON file produced
      by the retrieval stage and writes an enriched output file.

    Example:
        gen = Generator()
        gen.answer("What is KV cache?", long_context_string)
        gen.answer_dataset("data/search.json", "data/answers.json")
    """

    def __init__(self) -> None:
        self.llm = SmallLLM()
        self.chunker = Chunker(max_chunk_size=_CHUNK_SIZE)

    def build_prompt(self, question: str, context: str) -> str:
        """
        Assemble a fully-formatted few-shot prompt.

        Prepends the shared ``FEW_SHOT`` demonstration and appends the
        live context/question pair with a trailing "Answer:".

        Args:
            question: The natural-language question to answer.
            context: Retrieved context text (already truncated by the
                caller to fit within token budget).

        Returns:
            A complete prompt string ready for model inference.
        """
        return (
            FEW_SHOT
            + f"Context: {context}\n"
            + f"Question: {question}\n"
            + "Answer:"
        )

    def answer(self, question: str, context: str) -> None:
        """
        Answer a single question and print the result to stdout.

        Chunks ``context``, uses the first two chunks, builds a prompt,
        runs inference, and pretty-prints the question, answer, and
        elapsed time using Rich markup.

        Args:
            question: The natural-language question to answer.
            context: Raw context string (will be chunked internally).
        """
        chunks = self.chunker.chunk_text(context)

        # Fall back to the original context when chunking yields nothing
        # (e.g., very short inputs below the minimum chunk threshold).
        if not chunks:
            chunks = [context]

        # Concatenate the first two chunks to stay within token budget
        # while maximising the amount of evidence provided.
        combined_context = " ".join(chunks[:_NUM_CHUNKS])
        prompt = self.build_prompt(question, combined_context)

        raw, _, elapsed = self.llm.generate_batch([prompt])[0]

        result = (
            clean_answer(raw, question) if raw else "Not found in context."
        )
        if not result:
            result = "Not found in context."

        print(f"[bold cyan]Question:[/bold cyan] {question}")
        print(f"[bold green]Answer:[/bold green] {result}")
        print(f"[bold yellow]Time:[/bold yellow] {elapsed:.2f}s")

    def _prepare_batch(
        self, results_batch: list[MinimalSearchResults]
    ) -> list[str]:
        """
        Build a list of prompts from a batch of retrieval results.

        For each result the method iterates over its ``retrieved_sources``,
        filters out unusable entries (empty, too short, or raw-include
        directives).

        Args:
            results_batch: A slice of ``StudentSearchResult`` objects
                from the current processing batch.

        Returns:
            A list of prompt strings parallel to ``results_batch``.
            Entries for results with no usable sources are empty strings
            so that ``SmallLLM._run_batch`` can skip them cheaply.
        """
        prompts = []
        for result in results_batch:
            chunks: list[str] = []
            # sources = sorted(
            #     result.retrieved_sources, key=lambda s: s.rank or 9999
            # )
            for src in result.retrieved_sources:
                # Skip sources that are empty, too short to be
                # informative, or mkdocs snippet-include directives.
                if (
                    src.content
                    and src.content.strip()
                    and len(src.content.strip()) > 50
                    and not src.content.strip().startswith("# --8<--")
                ):
                    source_chunks = self.chunker.chunk_text(
                        src.content.strip()
                    )
                    chunks.extend(source_chunks[:2])

            if not chunks:
                prompts.append("")
                continue

            combined = " ".join(chunks)
            prompts.append(self.build_prompt(result.question_str, combined))

        return prompts

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str,
    ) -> None:
        """
        Run batch inference over a retrieval-results JSON file.

        Reads a ``StudentSearchResults`` JSON file, sorts questions for
        cache efficiency, processes them in batches of ``_BATCH_SIZE``,
        and writes a ``StudentSearchResultsAndAnswer`` JSON file.

        Slow answers (per-question average > ``_TIMEOUT_SECONDS``) are
        collected and printed as a summary after all batches complete.

        Args:
            student_search_results_path: Path to the input JSON file
                produced by the retrieval stage.
            save_directory: Destination path for the output JSON file.
                Parent directories are created automatically.
        """
        # 1. Load and validate the input data
        with open(student_search_results_path, "r") as fd:
            data = json.load(fd)

        search_data = StudentSearchResults.model_validate(data)
        all_results = search_data.search_results

        # Sort questions so that similar-length items are adjacent,
        # improving KV-cache reuse across consecutive prompts.
        all_results = _quick_sort_similarity(all_results)

        # 2. Initialise accumulators
        answers: list[MinimalAnswer] = []
        timings: list[float] = []
        slow_answers: list[dict[str, Any]] = []

        print(
            f"[bold green]Processing "
            f"{len(all_results)} questions.[/bold green]"
        )

        # Partition the full result list into fixed-size batches.
        batches = [
            all_results[i:i + _BATCH_SIZE]
            for i in range(0, len(all_results), _BATCH_SIZE)
        ]

        # 3. Process each batch
        progress_bar = bar(batches, color="green")
        for batch_idx, batch in enumerate(progress_bar):
            batch_start = time.perf_counter()

            # Build prompts and run the LLM over the whole batch.
            prompts = self._prepare_batch(batch)
            raw_answers = self.llm.generate_batch(prompts)

            batch_elapsed = time.perf_counter() - batch_start

            # Unpack per-question results and build MinimalAnswer objects.
            for i, result in enumerate(batch):
                raw, timed_out, elapsed = raw_answers[i]

                final_answer = (
                    clean_answer(raw, result.question_str)
                    if raw else "Not found in context."
                )
                if not final_answer:
                    final_answer = "Not found in context."

                timings.append(elapsed)

                if timed_out:
                    slow_answers.append({
                        "question": result.question_str,
                        "answer": final_answer,
                        "time": elapsed,
                    })

                answers.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question_str=result.question_str,
                        retrieved_sources=result.retrieved_sources,
                        answer=final_answer,
                    )
                )

            # Update the progress-bar description with timing stats.
            progress_bar.set_description(
                f"Batch {batch_idx} | "
                f"{batch_elapsed:.2f}s "
                f"({batch_elapsed / len(batch):.2f}s/q)"
            )

        # 4. Summarise and persist results
        total_time = sum(timings)
        print(
            f"\n[bold green]Total time: "
            f"{total_time:.2f}s[/bold green]"
        )

        _print_slow_answers(slow_answers)

        output_data = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_data.k
        )

        os.makedirs(os.path.dirname(save_directory), exist_ok=True)

        if save_directory.endswith("/") or os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
            file_path = os.path.join(save_directory, "dataset.json")
        else:
            os.makedirs(os.path.dirname(save_directory), exist_ok=True)
            file_path = save_directory

        with open(file_path, "w") as fd:
            json.dump(
                output_data.model_dump(
                    exclude={
                        "search_results": {
                            "__all__": {
                                "retrieved_sources": {
                                    "__all__": {"content"}
                                }
                            }
                        }
                    }
                ),
                fd, indent=4, ensure_ascii=False,
            )

        print(
            f"[bold green]Answers saved to:"
            f" {save_directory}[/bold green]"
        )
