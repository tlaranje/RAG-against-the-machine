"""Answer generation module using a small local LLM (Qwen3-0.6B)."""

from student.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
)

from student.utils import bar
from threading import Thread
from rich import print  # noqa: A001
from typing import Any

import torch as t
import time
import json
import os


class SmallLLM:
    """Wrapper around a small causal LLM loaded on the best available device.

    Args:
        model_name: HuggingFace model identifier.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Initialise tokeniser and model, move to the best device."""
        if t.cuda.is_available():
            self.device = t.device("cuda")
        else:
            self.device = t.device("cpu")

        self.dtype = (
            t.float16
            if self.device.type == "cuda"
            else t.float32
        )

        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_name
        )

        self.tokenizer.pad_token = (
            self.tokenizer.pad_token
            or self.tokenizer.eos_token
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )

        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            t.backends.cudnn.benchmark = True  # type: ignore

        print(
            f"[bold green]Loaded "
            f"{model_name} on "
            f"{self.device}[/bold green]"
        )


class Generator:
    """Batch answer generator using SmallLLM and BM25 search results."""

    def __init__(self) -> None:
        """Initialise the generator by loading the LLM."""
        self.llm = SmallLLM()

    @staticmethod
    def batch_questions(
        questions: list[MinimalSearchResults],
    ) -> list[list[MinimalSearchResults]]:
        """Split questions into sorted batches for efficient processing.

        Args:
            questions: List of search results with questions.

        Returns:
            List of batches, each containing up to 32 questions.
        """
        batch_size = 32

        questions = sorted(
            questions,
            key=lambda q: len(q.question),
        )

        return [
            questions[i:i + batch_size]
            for i in range(0, len(questions), batch_size)
        ]

    def build_prompt(
        self,
        result: MinimalSearchResults,
    ) -> str:
        """Build a chat prompt string from a search result.

        Args:
            result: A search result containing question and context.

        Returns:
            A formatted prompt string ready for tokenisation.
        """
        context = (
            result.content  # type: ignore[attr-defined]
            if hasattr(result, "content") and result.content  # type: ignore
            else "No context available."
        )

        context = context[:1000]

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer in 2 complete sentences. "
                    "Use only the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {result.question}"
                ),
            },
        ]

        return self.llm.tokenizer.apply_chat_template(  # type: ignore
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def answer_batch(
        self,
        batch: list[MinimalSearchResults],
        batch_idx: int,
    ) -> list[tuple[str, float]]:
        """Generate answers for a batch of questions.

        Args:
            batch: A list of search results to answer.
            batch_idx: Index of the current batch (used for progress display).

        Returns:
            A list of (answer_text, elapsed_per_item) tuples.
        """
        tokenizer = self.llm.tokenizer

        prompts = [
            self.build_prompt(result)
            for result in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=320,
            pad_to_multiple_of=8,
        ).to(self.llm.device)

        outputs: Any = None
        error: Exception | None = None

        def run_model() -> None:
            """Run the model in a background thread."""
            nonlocal outputs
            nonlocal error

            try:
                with t.inference_mode():
                    outputs = self.llm.model.generate(
                        **inputs,
                        max_new_tokens=40,
                        do_sample=False,
                        use_cache=True,
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except Exception as exc:
                error = exc

        start = time.perf_counter()

        thread = Thread(target=run_model)
        thread.start()

        with bar(
            total=None,
            desc=f"Batch {batch_idx}",
            position=1,
            leave=False,
            color="red",
        ) as pbar:
            while thread.is_alive():
                thread.join(timeout=0.05)
                pbar.update(1)

        thread.join()

        if error is not None:
            raise error

        if outputs is None:
            raise RuntimeError("Generation failed.")

        elapsed = time.perf_counter() - start
        answers: list[tuple[str, float]] = []

        for i, output in enumerate(outputs):
            input_len = int(
                inputs["attention_mask"][i].sum()
            )

            answer = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            )

            answers.append((
                " ".join(answer.split()),
                elapsed / len(batch),
            ))

        return answers

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers for all questions in a search results file.

        Args:
            student_search_results_path: Path to the JSON file produced
                by ``search_dataset``.
            save_directory: Directory where the output JSON will be saved.

        Raises:
            FileNotFoundError: If the input path does not exist.
            RuntimeError: If model generation fails.
        """
        with open(
            student_search_results_path,
            "r",
            encoding="utf-8",
        ) as fd:
            raw = json.load(fd)

        results = StudentSearchResults.model_validate(raw)
        total = len(results.search_results)
        batches = self.batch_questions(results.search_results)

        os.system("clear")

        print(
            f"[bold green]Loaded "
            f"{total} questions[/bold green]"
        )

        answers: list[MinimalAnswer] = []
        timings: list[tuple[float, str]] = []

        for batch_idx, batch in enumerate(
            bar(
                data=batches,
                desc="Generating answers",
                position=0,
                leave=True,
                color="green",
            ),
            start=1,
        ):
            batch_answers = self.answer_batch(batch, batch_idx)

            for result, (answer_text, elapsed) in zip(
                batch, batch_answers
            ):
                timings.append((elapsed, result.question))

                answers.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question=result.question,
                        retrieved_sources=result.retrieved_sources,
                        answer=answer_text,
                    )
                )

        print(
            f"[bold green]"
            f"Processed {total}/{total} questions"
            f"[/bold green]"
        )

        os.makedirs("data/output", exist_ok=True)

        with open(
            "data/output/time.txt",
            "w",
            encoding="utf-8",
        ) as fd:
            for rank, (elapsed, question) in enumerate(
                sorted(timings, key=lambda x: x[0]),
                start=1,
            ):
                truncated = (
                    question[:60] + "..."
                    if len(question) > 60
                    else question
                )
                fd.write(
                    f"{rank:>3}. {elapsed:6.2f}s {truncated}\n"
                )

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=results.k,
        )

        os.makedirs(save_directory, exist_ok=True)

        file_name = os.path.basename(student_search_results_path)
        output_path = os.path.join(save_directory, file_name)

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as fd:
            json.dump(
                output.model_dump(),
                fd,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"\n[bold green]"
            f"Saved results to:\n{output_path}"
            f"[/bold green]"
        )
