from student.utils import bar
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
        model_name = "Qwen/Qwen3-0.6b"

        # Verifica se o CUDA realmente está visível para este processo
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[bold yellow]Carregando {model_name} em: {device.upper()}[/bold yellow]")

        if device == "cpu":
            print("[bold red]AVISO: GPU não detectada! A geração será MUITO lenta.[/bold red]")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Usamos 'dtype' em vez de 'torch_dtype' para evitar o Warning que você recebeu
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # Se for CPU, usamos float32. Se for GPU, usamos float16.
            dtype=torch.float16 if device == "cuda" else torch.float32,
            # Se tiver GPU, joga lá. Se não, usa a CPU.
            device_map="auto" if device == "cuda" else None
        )

        if device == "cpu":
            self.model.to("cpu")

        print(f"[bold green]Modelo carregado com SUCESSO em {device.upper()}![/bold green]")

    def generate(self, prompt: str) -> str:
        # Move os dados para o mesmo hardware do modelo automaticamente
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=35,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_decoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        return full_text[len(prompt_decoded):].strip()


class Generator:
    def __init__(self) -> None:
        self.llm = SmallLLM()

    def _compress_and_clean(self, text: str) -> str:
        text = re.sub(r'#.*', '', text)
        text = re.sub(r'"{3}.*?"{3}', '', text, flags=re.DOTALL)
        return " ".join(text.split())

    def build_prompt(self, question: str, context: str) -> str:
        # Prompt otimizado para modelos Instruct
        few_shot = (
            "Context: class VllmConfig: Dataclass for all vllm configuration.\n"
            "Question: What is the main configuration object in vLLM?\n"
            "Answer: The VllmConfig dataclass.\n\n"
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

        for result in progress_bar:
            t0 = time.perf_counter()

            clean_ctx = self._compress_and_clean(result.content)
            prompt = self.build_prompt(result.question, clean_ctx)

            # Chama a geração do modelo Transformers
            raw_text = self.llm.generate(prompt)
            final_answer = clean_answer(raw_text)

            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            if elapsed > 2.0:
                slow_responses += 1

            progress_bar.set_description(f"Last: {elapsed:.2f}s")

            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=final_answer
            ))

        # --- Métricas de Saída ---
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
                output_data.model_dump(), fd, indent=4, ensure_ascii=False
            )
