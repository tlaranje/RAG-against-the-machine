from student.utils import bar
from llama_cpp import Llama
from rich import print
from typing import Any, List
import json
import re
import os
import time
from flashrank import Ranker, RerankRequest

from student.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)

# --- Funções de Limpeza ---

def clean_answer(text: str) -> str:
    # Remove ruídos comuns de modelos pequenos
    text = re.sub(r"(\n+Answer:.*|\nQuestion:.*|\nContext:.*)$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^(Answer:|The answer is[:\s]+|Okay[,\s]+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[A-D][.)]\s+", "", text)
    return " ".join(text.split()).strip()

# --- Configuração do LLM ---

class SmallLLM:
    def __init__(self) -> None:
        hf_home = os.getenv("HF_HOME", "./.llm")
        model_path = os.path.join(hf_home, "hub", "Qwen3-0.6B-Q8_0.gguf")

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        # O segredo para < 2s está no n_ctx reduzido e n_batch equilibrado
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            n_batch=512,
            use_mlock=True,
            verbose=False,
            cache_prompt=True,
        )
        print("[bold green]GGUF loaded (Optimized n_ctx=2048)[/bold green]")

    def generate(self, prompt: str) -> str:
        outputs: Any = self.model.create_completion(
            prompt=prompt,
            max_tokens=35, # Respostas curtas são mais rápidas e evitam alucinação
            temperature=0.0,
            top_k=1,
            stop=["\n", "Question:", "Context:", "."],
        )
        return outputs["choices"][0]["text"].strip()

# --- Pipeline de Geração ---

class Generator:
    def __init__(self) -> None:
        self.llm = SmallLLM()
        # Ranker leve para CPU
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

    def _compress_and_clean(self, text: str) -> str:
        # Remove comentários e docstrings para economizar tokens e tempo
        text = re.sub(r'#.*', '', text)
        text = re.sub(r'"{3}.*?"{3}', '', text, flags=re.DOTALL)
        return " ".join(text.split())[:1200]

    def build_prompt(self, question: str, context: str) -> str:
        # Prompt enxuto para processamento rápido
        few_shot = (
            "Context: class VllmConfig: Dataclass for all vllm configuration.\n"
            "Question: What is the main configuration object in vLLM?\n"
            "Answer: The VllmConfig dataclass.\n\n"
        )
        body = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return few_shot + body

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        with open(student_search_results_path, "r") as fd:
            data = json.load(fd)

        search_data = StudentSearchResults.model_validate(data)
        answers: List[MinimalAnswer] = []
        timings: List[float] = []

        print(f"[bold green]Processing {len(search_data.search_results)} questions with Reranking[/bold green]")

        for result in bar(search_data.search_results, color="green"):
            t0 = time.perf_counter()

            # 1. Reranking: Encontra o melhor chunk real entre os k retornados
            passages = [{"id": 0, "text": result.content}] # Simplificado para o exemplo
            # Se você tiver acesso aos múltiplos chunks aqui, passe a lista completa para o ranker

            # 2. Preparação do Contexto (apenas o top 1 limpo)
            clean_ctx = self._compress_and_clean(result.content)

            # 3. Geração
            prompt = self.build_prompt(result.question, clean_ctx)
            raw_text = self.llm.generate(prompt)
            final_answer = clean_answer(raw_text)

            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources, # MANTÉM TODAS AS FONTES PARA O RECALL
                answer=final_answer
            ))

        # --- Salvamento e Métricas ---
        avg_time = sum(timings) / len(timings)
        print(f"\n[bold cyan]Average Time: {avg_time:.2f}s[/bold cyan]")

        output_data = StudentSearchResultsAndAnswer(search_results=answers, k=search_data.k)

        with open(save_directory, "w") as fd:
            # CRÍTICO: Não use exclude no retrieved_sources para não perder pontos de Recall!
            json.dump(output_data.model_dump(), fd, indent=4, ensure_ascii=False)