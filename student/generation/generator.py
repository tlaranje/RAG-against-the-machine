from student.utils import bar
from llama_cpp import Llama
from rich import print
from typing import Any, List
import json
import re
import os
import time

from student.models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


def clean_answer(text: str) -> str:
    text = re.sub(
        r"(\n+Answer:.*|\nQuestion:.*|\nContext:.*)$",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"^(Answer:|The answer is[:\s]+|Okay[,\s]+)",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"^[A-D][.)]\s+", "", text)
    return " ".join(text.split()).strip()


class SmallLLM:
    def __init__(self) -> None:
        hf_home = os.getenv("HF_HOME", "./.llm")
        model_path = os.path.join(hf_home, "hub", "Qwen")

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            n_batch=512,
            use_mlock=True,
            verbose=False,
            cache_prompt=True,
        )

    def generate(self, prompt: str) -> str:
        outputs: Any = self.model.create_completion(
            prompt=prompt,
            max_tokens=35,
            temperature=0.0,
            top_k=1,
            stop=["\n", "Question:", "Context:", ". "],
        )
        return outputs["choices"][0]["text"].strip()


class Generator:
    def __init__(self) -> None:
        self.llm = SmallLLM()
        self.re_comments = re.compile(r'#[^\n]*')
        self.re_docstrings = re.compile(r'"{3}.*?"{3}', re.DOTALL)

    def _compress_and_clean(self, text: str) -> str:
        text = self.re_comments.sub('', text)
        text = self.re_docstrings.sub('', text)
        return " ".join(text.split())

    def build_prompt(self, question: str, context: str) -> str:
        few_shot = (
            "Context: class VllmConfig:"
            " Dataclass for all vllm configuration.\n"
            "Question: What is the main configuration object in vLLM?\n"
            "Answer: VllmConfig.\n\n"
        )
        body = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return few_shot + body

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        with open(student_search_results_path, "r") as fd:
            data = json.load(fd)

        search_data = StudentSearchResults.model_validate(data)
        answers: List[MinimalAnswer] = []
        timings: List[float] = []
        slow_responses = 0

        print(
            f"[bold green]Processing {len(search_data.search_results)} "
            "questions[/bold green]"
        )

        progress_bar = bar(search_data.search_results, color="green")
        i = 0
        for result in progress_bar:
            t0 = time.perf_counter()

            if result.retrieved_sources:
                top_1_source = result.retrieved_sources[0]
                contexto = top_1_source.content if top_1_source.content else ""
            else:
                contexto = ""

            clean_ctx = self._compress_and_clean(contexto)

            prompt = self.build_prompt(result.question, clean_ctx)
            raw_text = self.llm.generate(prompt)
            final_answer = clean_answer(raw_text)

            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            if elapsed > 2.0:
                slow_responses += 1

            progress_bar.set_description(
                f"Question {i} | Prev: {elapsed:.2f}s"
            )

            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=final_answer
            ))
            i += 1

        print(f"[bold green]Total time: {sum(timings):.2f}s[/bold green]")
        print(f"[bold green]Answers save on: {save_directory}[/bold green]")

        avg_time = sum(timings) / len(timings)
        max_time = max(timings)
        print("\n" + "─" * 50)
        print("[bold cyan]Métricas de Performance (GPU):[/bold cyan]")
        print(f"Média de Tempo: [yellow]{avg_time:.2f}s[/yellow]")
        print(f"Tempo Máximo:  [yellow]{max_time:.2f}s[/yellow]")
        print(
            f"Acima de 2.0s: [red]{slow_responses}[/red] de {len(timings)} "
            f"({(slow_responses/len(timings))*100:.1f}%)"
        )
        print("─" * 50)

        output_data = StudentSearchResultsAndAnswer(
            search_results=answers, k=search_data.k
        )

        with open(save_directory, "w") as fd:
            json.dump(
                output_data.model_dump(
                    exclude={
                        "search_results": {"__all__": {
                            "content", "retrieved_sources"
                        }}
                    }
                ), fd, indent=4, ensure_ascii=False
            )
