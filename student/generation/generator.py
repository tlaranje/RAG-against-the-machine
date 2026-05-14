from student.utils import bar
from llama_cpp import Llama
from rich import print
from typing import Any
import json
import re
import os
import time

from student.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)

# ---------------------------------------------------------
# Regex cleanup
# ---------------------------------------------------------

_NOISE_PREFIX = re.compile(
    r"^(Answer:|The answer is[:\s]+|Okay[,\s]+|Let me[^.]*\.\s*)",
    re.IGNORECASE,
)

_NOISE_SUFFIX = re.compile(
    r"(\n+Answer:.*|\nQuestion:.*|\nContext:.*)$",
    re.DOTALL | re.IGNORECASE,
)

_MULTIPLE_CHOICE = re.compile(r"^[A-D][.)]\s+")


def clean_answer(text: str) -> str:
    text = text.strip()

    text = _NOISE_SUFFIX.sub("", text)

    for _ in range(2):
        new = _NOISE_PREFIX.sub("", text).strip()
        if new == text:
            break
        text = new

    text = _MULTIPLE_CHOICE.sub("", text).strip()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Few-shot
# ---------------------------------------------------------

_FEW_SHOT = """Context: The /v1/load_lora_adapter POST endpoint is used to dynamically load LoRA adapters.
Question: What HTTP endpoint loads a LoRA adapter in vLLM?
Answer: The POST /v1/load_lora_adapter endpoint.

Context: vLLM exposes production metrics via a Prometheus endpoint at /metrics.
Question: What endpoint does vLLM use to expose production metrics?
Answer: The /metrics endpoint.

"""


# ---------------------------------------------------------
# Small LLM
# ---------------------------------------------------------

class SmallLLM:
    def __init__(self) -> None:

        hf_home = os.getenv("HF_HOME", "./.llm")

        model_path = os.path.join(
            hf_home,
            "hub",
            "Qwen3-0.6B-Q8_0.gguf"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        threads = max(1, (os.cpu_count() or 4) // 2)

        self.model = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=threads,
            n_threads_batch=threads,
            n_batch=256,
            use_mlock=True,
            verbose=False,
            cache_prompt=True,
        )

        print("[bold green]GGUF loaded[/bold green]")

    def generate(self, prompt: str) -> str:

        outputs: Any = self.model.create_completion(
            prompt=prompt,

            max_tokens=24,
            temperature=0.0,

            top_k=1,
            top_p=0.95,

            repeat_penalty=1.15,

            stop=[
                "\n",
                "Question:",
                "Context:",
            ],
        )

        return outputs["choices"][0]["text"].strip()


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

class Generator:
    def __init__(self) -> None:
        self.llm = SmallLLM()

    # -----------------------------------------------------
    # Prompt Builder
    # -----------------------------------------------------

    def build_prompt(
        self,
        result: MinimalSearchResults
    ) -> str:

        question = re.sub(
            r"_{2,}",
            "what",
            result.question
        )

        # IMPORTANT:
        # Smaller context = MUCH faster
        clean_ctx = ""

        if result.content and len(result.content.strip()) > 20:

            clean_ctx = " ".join(
                result.content[:700].split()
            )

        if clean_ctx:
            body = (
                f"Context: {clean_ctx}\n"
                f"Question: {question}\n"
            )
        else:
            body = f"Question: {question}\n"

        return f"{_FEW_SHOT}{body}Answer:"

    # -----------------------------------------------------
    # Single Answer
    # -----------------------------------------------------

    def answer(
        self,
        question: str,
        context: list[str] | str
    ) -> None:

        if isinstance(context, list):
            context = "\n".join(context)

        mock = MinimalSearchResults(
            question_id="single",
            question=question,
            retrieved_sources=[],
            content=context,
        )

        t0 = time.perf_counter()

        raw = self.llm.generate(
            self.build_prompt(mock)
        )

        elapsed = time.perf_counter() - t0

        answer_text = clean_answer(raw)

        output = {
            "question": question,
            "answer": answer_text
        }

        os.makedirs("data/output", exist_ok=True)

        with open(
            "data/output/single_answer.json",
            "w",
            encoding="utf-8"
        ) as fd:

            json.dump(
                output,
                fd,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"[green]Saved[/green] "
            f"({elapsed:.2f}s)"
        )

    # -----------------------------------------------------
    # Dataset Answering
    # -----------------------------------------------------

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:

        with open(
            student_search_results_path,
            "r",
            encoding="utf-8"
        ) as fd:

            raw = json.load(fd)

        results = StudentSearchResults.model_validate(raw)

        total = len(results.search_results)

        print(
            f"[bold green]Processing {total} questions[/bold green]"
        )

        answers: list[MinimalAnswer] = []

        timings: list[float] = []

        pbar = bar(
            results.search_results,
            color="green",
            position=0
        )

        for i, result in enumerate(pbar, start=1):

            t0 = time.perf_counter()

            prompt = self.build_prompt(result)

            raw_text = self.llm.generate(prompt)

            answer_text = clean_answer(raw_text)

            elapsed = time.perf_counter() - t0

            timings.append(elapsed)

            pbar.set_description(
                f"[{i}/{total}] {elapsed:.2f}s"
            )

            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        # -------------------------------------------------
        # Timing
        # -------------------------------------------------

        total_s = sum(timings)

        avg_s = total_s / len(timings)

        max_s = max(timings)

        min_s = min(timings)

        over_2s = sum(
            1 for t in timings if t > 2.0
        )

        print("\n[bold cyan]━━━ Timing Summary ━━━[/bold cyan]")

        print(f"Total : {total}")
        print(f"Total time : {total_s:.1f}s")
        print(f"Average : {avg_s:.2f}s")
        print(f"Min/Max : {min_s:.2f}s / {max_s:.2f}s")
        print(f">2s : {over_2s}")

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        output_data = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=results.k,
        )

        os.makedirs(save_directory, exist_ok=True)

        output_path = os.path.join(
            save_directory,
            os.path.basename(student_search_results_path)
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as fd:

            json.dump(
                output_data.model_dump(
                    exclude={
                        "search_results": {
                            "__all__": {
                                "content",
                                "retrieved_sources"
                            }
                        }
                    }
                ),
                fd,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"\n[bold green]Saved:[/bold green] {output_path}"
        )
