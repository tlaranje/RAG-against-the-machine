from student.utils import bar
from threading import Thread
from llama_cpp import Llama
from rich import print
from typing import Any
import json
import re
import os

from student.models import (
    MinimalAnswer, MinimalSearchResults,
    StudentSearchResults, StudentSearchResultsAndAnswer
)


def clean_answer(text: str) -> str:
    # Remove blocos de código e artefactos comuns
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = text.replace("\\boxed{", "").replace("}", "")
    text = text.replace('"""', "").replace("```", "").strip()

    # Remove prefill se repetido
    if text.startswith("Answer:"):
        text = text[len("Answer:"):].strip()

    # Corta tudo após um segundo "Answer:" ou "Question:"
    for noise in ["Answer:", "Question:", "Context:", "\nAnswer", "####"]:
        if noise in text:
            text = text[:text.index(noise)].strip()

    # Remove multiple-choice artefactos
    if text.startswith(("A.", "B.", "C.", "D.")):
        text = text[2:].strip()

    # Remove frase final se termina em ":"  — estava incompleta
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)]
    sentences = [s for s in sentences if s and not s.endswith(":")]
    text = " ".join(sentences)

    # Muito curto = inútil
    if len(text) < 8:
        return "No information found in the provided context."

    return text.strip()


class SmallLLM:
    def __init__(
        self, model_path: str = "models/qwen3-0.6b-q4_k_m.gguf"
    ) -> None:
        if not os.path.exists(model_path):
            print(
                "[bold red]Arquivo GGUF não "
                "encontrado em: {model_path}[/bold red]"
            )
            raise FileNotFoundError(model_path)

        # `n_ctx` sets the maximum context window (tokens).
        # `n_threads` maps to the number of CPU threads used for matrix
        # `n_batch` controls how many tokens are processed in one forward pass
        # `cache_prompt` reuses the KV-cache for the system prompt across
        # multiple calls, avoiding redundant computation.

        physical_cores = os.cpu_count()
        assert physical_cores is not None
        use_threads = max(1, physical_cores - 1)

        self.model = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=use_threads,
            n_threads_batch=use_threads,
            n_batch=128,
            verbose=False,
            cache_prompt=True,
        )
        print("[bold green]Modelo GGUF carregado via CPU[/bold green]")

    def generate(self, prompt: str, i: int = 1) -> str:
        outputs: Any = None

        def run_model() -> None:
            nonlocal outputs
            outputs = self.model.create_completion(
                prompt=prompt,
                max_tokens=80,
                temperature=0,
                repeat_penalty=1.3,
                stop=["Question:", "Context:", "\n\n"],
            )

        thread = Thread(target=run_model)
        thread.start()

        with bar(
            prompt, desc=f"Answering question {i}",
            color="red", total=100, position=1
        ) as pbar:
            while thread.is_alive():
                thread.join(timeout=0.012)
                pbar.update(1)

        content = outputs["choices"][0]["text"].strip()

        noise_patterns = ["Okay,", "Let me", "Answer:", "The answer is", "```"]
        for pattern in noise_patterns:
            if content.startswith(pattern):
                content = content.replace(pattern, "", 1).strip()

        if "A)" in content or "1." in content:
            content = content.split("\n")[0]

        return content if len(content) > 2 else "No information found."


class Generator:
    def __init__(
        self, model_path: str = "models/qwen3-0.6b-q4_k_m.gguf"
    ) -> None:
        self.llm = SmallLLM(model_path)

    def build_prompt(self, result: MinimalSearchResults) -> str:
        context = result.content[:500] if result.content else "No information."

        # Dois exemplos concretos ensinam o padrão ao modelo
        few_shot = (
            "Context: The Eiffel Tower is located in Paris, France. "
            "It was built in 1889.\n"
            "Question: Where is the Eiffel Tower located?\n"
            "Answer: The Eiffel Tower is located in Paris, France.\n\n"

            "Context: vLLM uses PagedAttention to manage memory efficiently. "
            "It supports NVIDIA and AMD GPUs.\n"
            "Question: What memory management technique does vLLM use?\n"
            "Answer: vLLM uses PagedAttention to manage memory efficiently.\n\n"
        )

        return (
            f"{few_shot}"
            f"Context: {context}\n"
            f"Question: {result.question}\n"
            f"Answer:"
        )

    def answer(self, question: str, context: list[str] | str) -> None:
        if isinstance(context, list):
            context = "\n".join(context)

        res_mock = MinimalSearchResults(
            question_id="single", question=question,
            retrieved_sources=[], content=context
        )
        answer_text = clean_answer(
            self.llm.generate(self.build_prompt(res_mock))
        )
        output = {
            "question": question, "answer": answer_text, "context": context
        }
        os.makedirs("data/output", exist_ok=True)
        with open("data/output/single_answer.json", "w") as fd:
            json.dump(output, fd, indent=4, ensure_ascii=False)

        print("[bold green]Resposta única salva.[/bold green]")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str,
    ) -> None:
        with open(student_search_results_path, "r", encoding="utf-8") as fd:
            raw = json.load(fd)

        results = StudentSearchResults.model_validate(raw)
        total = len(results.search_results)

        os.system("clear")
        print(f"[bold green]Processando {total} questões ...[/bold green]")

        answers: list[MinimalAnswer] = []
        for i, result in enumerate(bar(
            results.search_results, desc="Generating answers",
            color="green", position=0
        ), start=1):
            messages = self.build_prompt(result)
            answer_text = self.llm.generate(messages, i)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        output_data = StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k,
        )
        os.makedirs(save_directory, exist_ok=True)

        file_name = os.path.basename(student_search_results_path)
        output_path = os.path.join(save_directory, file_name)

        with open(output_path, "w", encoding="utf-8") as fd:
            json.dump(
                output_data.model_dump(
                    exclude={"search_results": {"__all__": {"content", "retrieved_sources"}}}
                ),
                fd,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"\n[bold green]Sucesso! Resultados em: {output_path}[/bold green]"
        )
