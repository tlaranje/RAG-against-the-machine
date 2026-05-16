from student.utils import bar
from llama_cpp import Llama
from rich import print
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import os
import time

from student.models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)

_MAX_CTX_CHARS = 600
_MAX_NEW_TOKENS = 45
_RERANKER_TOP_K = 3
_BATCH_SIZE = 8
_N_THREADS = os.cpu_count() or 4
_THREADS_PER_WORKER = max(1, _N_THREADS // _BATCH_SIZE)

# Few-shot cobre os padrões de resposta mais comuns:
# valor simples, lista, flag CLI, numérico, negativo, limitação
FEW_SHOT = (
    # Parâmetro/valor simples
    "Context: vLLM sets linear_method according to different"
    " quantization schemes to support weight quantization in linear layers.\n"
    "Question: What parameter does vLLM set for weight quantization?\n"
    "Answer: linear_method\n\n"

    # Variável de ambiente
    "Context: You can manually set the attention backend by configuring"
    " the environment variable VLLM_ATTENTION_BACKEND.\n"
    "Question: How can you manually set the attention backend in vLLM?\n"
    "Answer: By configuring the environment variable VLLM_ATTENTION_BACKEND\n\n"

    # Endpoint
    "Context: vLLM exposes production metrics at the /metrics endpoint.\n"
    "Question: What endpoint does vLLM use to expose production metrics?\n"
    "Answer: /metrics\n\n"

    # Lista de três itens
    "Context: The three key abstractions used for disaggregated prefilling"
    " in vLLM are: KV pipe, KV lookup buffer, and KV connector.\n"
    "Question: What are the three key abstractions used for disaggregated"
    " prefilling in vLLM?\n"
    "Answer: KV pipe, KV lookup buffer, and KV connector\n\n"

    # Paralelismo — evita confundir TP com EP
    "Context: vLLM supports Expert Parallelism for large-scale deployment"
    " of Mixture of Experts models.\n"
    "Question: What parallelism strategy does vLLM support for large-scale"
    " deployment of Mixture of Experts models?\n"
    "Answer: Expert Parallelism\n\n"

    # Numérico
    "Context: Each warp needs 8 inner iterations to handle a whole block"
    " of value tokens when BLOCK_SIZE=16, V_VEC_SIZE=8, HEAD_SIZE=128,"
    " WARP_SIZE=32.\n"
    "Question: How many inner iterations does a warp need to handle a whole"
    " block of value tokens when BLOCK_SIZE is 16, V_VEC_SIZE is 8,"
    " HEAD_SIZE is 128, and WARP_SIZE is 32?\n"
    "Answer: 8\n\n"

    # Limitação/restrição
    "Context: MLP speculators are not compatible with pipeline parallelism.\n"
    "Question: What is the current limitation when using MLP speculators"
    " for speculative decoding in vLLM?\n"
    "Answer: Not compatible with pipeline parallelism\n\n"

    # Lista de backends técnicos
    "Context: The three communication backends available for Expert"
    " Parallelism in vLLM are DeepEP, PPLX, and PyNCCL.\n"
    "Question: What are the three communication backends available for"
    " Expert Parallelism in vLLM?\n"
    "Answer: DeepEP, PPLX, and PyNCCL\n\n"

    # Sim/não
    "Context: vLLM cannot serve multiple models on a single port using"
    " the OpenAI API.\n"
    "Question: Can vLLM serve multiple models on a single port using"
    " the OpenAI API?\n"
    "Answer: No\n\n"

    # Flag CLI
    "Context: The --reasoning-parser flag must be specified when serving"
    " a reasoning model in vLLM to extract reasoning content.\n"
    "Question: What flag do you need to specify when serving a reasoning"
    " model in vLLM to extract reasoning content?\n"
    "Answer: --reasoning-parser\n\n"
)


def clean_answer(text: str) -> str:
    """Clean generated answer removing artifacts and repetitions."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(
        r"(\nAnswer:.*|\nQuestion:.*|\nContext:.*)$",
        "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"^(Answer\s*:\s*|The answer is[:\s]+|Okay[,\s]+)",
        "", text, flags=re.IGNORECASE
    )
    return " ".join(text.split()).strip()


class Reranker:
    """Keyword overlap reranker — zero dependencies, ~0ms per question."""

    def _keyword_score(self, question: str, content: str) -> float:
        stopwords = {"the", "a", "an", "is", "in", "of", "to", "and",
                     "for", "how", "what", "does", "do", "can", "you",
                     "it", "with", "on", "are", "be", "by", "or", "at"}
        q_words = {
            w.lower() for w in re.findall(r'\w+', question)
            if w.lower() not in stopwords and len(w) > 2
        }
        c_words = {
            w.lower() for w in re.findall(r'\w+', content)
            if w.lower() not in stopwords and len(w) > 2
        }
        if not q_words:
            return 0.0
        return len(q_words & c_words) / len(q_words)

    def rerank(self, question: str, sources: list, top_k: int = _RERANKER_TOP_K) -> list:
        if not sources:
            return []
        scored = []
        for src in sources:
            content = getattr(src, "content", "") or ""
            content = " ".join(content.split())
            score = self._keyword_score(question, content) if content else 0.0
            scored.append((score, src))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [src for _, src in scored[:top_k]]


class SmallLLM:
    """Pool de instâncias Llama para processamento paralelo em CPU."""

    def __init__(self, n_workers: int = _BATCH_SIZE) -> None:
        hf_home = os.getenv("HF_HOME", "./.llm")
        model_path = os.path.join(hf_home, "hub", "Qwen")

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        print(f"[bold yellow]Loading {n_workers} model instances...[/bold yellow]")
        self._pool: List[Llama] = [
            Llama(
                model_path=model_path,
                n_ctx=1024,
                n_threads=_THREADS_PER_WORKER,
                n_batch=256,
                use_mlock=True,
                verbose=False,
                cache_prompt=True,
            )
            for _ in range(n_workers)
        ]
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        print(f"[bold green]{n_workers} instances ready "
              f"({_THREADS_PER_WORKER} threads each).[/bold green]")

    def warmup(self, prefix: str) -> None:
        print(
            "[bold yellow]Warming up KV cache on all workers...[/bold yellow]"
        )
        futures = {
            self._executor.submit(self._run_warmup, idx, prefix): idx
            for idx in range(len(self._pool))
        }
        for future in as_completed(futures):
            future.result()  # propaga excepções
        print(
            "[bold green]KV cache warm — few-shot "
            "cached on all workers.[/bold green]"
        )

    def _run_warmup(self, worker_idx: int, prefix: str) -> None:
        self._pool[worker_idx].create_completion(
            prompt=prefix,
            max_tokens=1,
            temperature=0.0,
        )

    def _run(self, worker_idx: int, prompt: str) -> str:
        model = self._pool[worker_idx]
        outputs: Any = model.create_completion(
            prompt=prompt,
            max_tokens=_MAX_NEW_TOKENS,
            temperature=0.0,
            top_k=1,
            stop=["\n", "Question:", "Context:"],
        )
        return outputs["choices"][0]["text"].strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        results = [""] * len(prompts)
        futures = {}
        for idx, prompt in enumerate(prompts):
            worker_idx = idx % len(self._pool)
            future = self._executor.submit(self._run, worker_idx, prompt)
            futures[future] = idx
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"[red]Worker error on prompt {idx}: {exc}[/red]")
                results[idx] = ""
        return results


class Generator:
    def __init__(self) -> None:
        self.reranker = Reranker()
        self.llm = SmallLLM(n_workers=_BATCH_SIZE)
        self.llm.warmup(FEW_SHOT)

    def _get_context(self, sources: list) -> str:
        parts = []
        total = 0
        for src in sources:
            content = getattr(src, "content", "") or ""
            content = " ".join(content.split())
            if not content:
                continue
            remaining = _MAX_CTX_CHARS - total
            if remaining <= 50:
                break
            chunk = content[:remaining]
            parts.append(chunk)
            total += len(chunk)
        return " ".join(parts)

    def build_prompt(self, question: str, context: str) -> str:
        ctx = context[:_MAX_CTX_CHARS]
        body = f"Context: {ctx}\nQuestion: {question}\nAnswer:"
        return FEW_SHOT + body

    def _prepare_batch(self, results_batch: list) -> tuple:
        prompts = []
        for result in results_batch:
            ranked = self.reranker.rerank(
                result.question,
                result.retrieved_sources or []
            )
            ctx = self._get_context(ranked)
            prompt = self.build_prompt(result.question, ctx) if ctx else ""
            prompts.append(prompt)
        return prompts

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        with open(student_search_results_path, "r") as fd:
            data = json.load(fd)

        search_data = StudentSearchResults.model_validate(data)
        all_results = search_data.search_results
        answers: List[MinimalAnswer] = []
        timings: List[float] = []
        slow_responses = 0

        print(
            f"[bold green]Processing {len(all_results)} questions "
            f"in batches of {_BATCH_SIZE}[/bold green]"
        )

        batches = [
            all_results[i: i + _BATCH_SIZE]
            for i in range(0, len(all_results), _BATCH_SIZE)
        ]

        progress_bar = bar(batches, color="green")
        for batch_idx, batch in enumerate(progress_bar):
            t0 = time.perf_counter()

            prompts = self._prepare_batch(batch)

            active_indices = [i for i, p in enumerate(prompts) if p]
            active_prompts = [prompts[i] for i in active_indices]

            raw_answers = [""] * len(batch)
            if active_prompts:
                generated = self.llm.generate_batch(active_prompts)
                for pos, idx in enumerate(active_indices):
                    raw_answers[idx] = generated[pos]

            batch_elapsed = time.perf_counter() - t0

            for i, result in enumerate(batch):
                raw = raw_answers[i]
                final_answer = clean_answer(raw) if raw else "Not found in context."
                if not final_answer:
                    final_answer = "Not found in context."

                per_q = batch_elapsed / len(batch)
                timings.append(per_q)
                if per_q > 2.0:
                    slow_responses += 1

                answers.append(MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=final_answer
                ))

            progress_bar.set_description(
                f"Batch {batch_idx} | {batch_elapsed:.2f}s "
                f"({batch_elapsed/len(batch):.2f}s/q)"
            )

        avg_time = sum(timings) / len(timings)
        max_time = max(timings)

        print(f"[bold green]Total time: {sum(timings):.2f}s[/bold green]")
        print(f"[bold green]Answers saved to: {save_directory}[/bold green]")
        print("\n" + "─" * 50)
        print("[bold cyan]Performance Metrics:[/bold cyan]")
        print(f"Average time: [yellow]{avg_time:.2f}s[/yellow]")
        print(f"Max time:     [yellow]{max_time:.2f}s[/yellow]")
        print(
            f"Above 2.0s:   [red]{slow_responses}[/red] de {len(timings)} "
            f"({(slow_responses / len(timings)) * 100:.1f}%)"
        )
        print("─" * 50)

        output_data = StudentSearchResultsAndAnswer(
            search_results=answers, k=search_data.k
        )

        os.makedirs(os.path.dirname(save_directory), exist_ok=True)

        with open(save_directory, "w") as fd:
            json.dump(
                output_data.model_dump(
                    exclude={
                        "search_results": {
                            "__all__": {"content", "retrieved_sources"}
                        }
                    }
                ),
                fd,
                indent=4,
                ensure_ascii=False
            )