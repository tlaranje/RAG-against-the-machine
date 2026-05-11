from student.models import (
    MinimalSource, MinimalAnswer, StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from student.utils import bar
from threading import Thread
from rich import print
from typing import Any
import torch as t
import json
import os
import time


class SmallLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.device = t.device(
            "mps" if t.backends.mps.is_available()
            else "cuda" if t.cuda.is_available()
            else "cpu"
        )
        dtype = (
            t.float16
            if self.device.type in ("cuda", "mps")
            else t.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype
        )
        # self.model = t.compile(self.model)
        self.model.to(self.device)
        self.model.eval()


class Generator:
    def __init__(self) -> None:
        self.llm = SmallLLM()

    def answer(
        self, prompt: str, sources: list[MinimalSource]
    ) -> tuple[str, float]:
        context_parts = []
        for s in sources[:1]:
            try:
                with open(s.file_path, "r") as fd:
                    fd.seek(s.first_character_index)
                    content = fd.read(min(
                        s.last_character_index - s.first_character_index,
                        200
                    ))
                context_parts.append(f"Source ({s.file_path}):\n{content}")
            except Exception:
                continue
        context = "\n\n".join(context_parts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer in plain English, no code, no lists. "
                    "One or two sentences maximum."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {prompt}"
            }
        ]
        tokenizer = self.llm.tokenizer
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.llm.device)

        output = []
        start_time = time.perf_counter()

        def run_model():
            nonlocal output
            with t.no_grad():
                output = self.llm.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    use_cache=True,
                    do_sample=False,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id
                )

        thread = Thread(target=run_model)
        thread.start()
        with bar(
            total=100, desc="Generating", position=1, leave=False, color="red"
        ) as pbar:
            for _ in range(100):
                while thread.is_alive():
                    thread.join(timeout=0.02)
                    if not thread.is_alive():
                        break
                    pbar.update(1)
        thread.join()

        elapsed = time.perf_counter() - start_time

        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        tokenizer = self.llm.tokenizer
        tokenizer.padding_side = "left"

        with open(student_search_results_path, "r") as fd:
            raw = json.load(fd)
        results = StudentSearchResults.model_validate(raw)
        total = len(results.search_results)
        os.system("clear")
        print(
            f"[bold green]Loaded {total} questions from "
            f"{student_search_results_path}[/bold green]"
        )

        answers = []
        timings: list[tuple[float, str]] = []  # (elapsed, question)

        for result in bar(
            data=results.search_results,
            desc="Generating answers",
            position=0,
            leave=True,
            color="green"
        ):
            answer_text, elapsed = self.answer(
                result.question,
                result.retrieved_sources
            )
            timings.append((elapsed, result.question))
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text
            ))

        print(f"[bold green]Processed {total}/{total} questions.[/bold green]")

        # Print timings sorted by elapsed time (ascending)
        print("\n[bold cyan]Answer times (fastest → slowest):[/bold cyan]")
        for rank, (elapsed, question) in enumerate(sorted(timings, key=lambda x: x[0]), 1):
            truncated = question[:60] + "..." if len(question) > 60 else question
            print(f"  [cyan]{rank:>3}.[/cyan] [white]{elapsed:6.2f}s[/white]  {truncated}")

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=results.k
        )
        os.makedirs(save_directory, exist_ok=True)
        file_name = student_search_results_path.rsplit("/", 1)[1]
        with open(f"{save_directory}/{file_name}", "w") as fd:
            json.dump(output.model_dump(), fd, indent=4)
        print(
            f"\n[bold green]Saved results to {save_directory}/"
            f"{file_name}[/bold green]"
        )
