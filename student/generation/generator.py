from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr
from student.utils import bar
from typing import Any, List
from llama_cpp import Llama
from rich import print
import time
import json
import re
import os

from student.models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)

_TOP_K_CHUNKS = 1
_MAX_CHUNK_CHARS = 300
_MAX_NEW_TOKENS = 45
_BATCH_SIZE = 8
_N_THREADS = os.cpu_count() or 4
_THREADS_PER_WORKER = max(1, _N_THREADS // _BATCH_SIZE)

FEW_SHOT = (
    "Instructions: Answer using ONLY the context. Be direct. No preamble."
    " No markdown. No URLs. If it is a list use commas."
    " If not in context: 'Not found in context'.\n\n"

    "Context: vLLM sets linear_method according to different"
    " quantization schemes to support weight quantization in linear layers.\n"
    "Question: What parameter does vLLM set for weight quantization?\n"
    "Answer: linear_method\n\n"

    "Context: You can manually set the attention backend by configuring"
    " the environment variable VLLM_ATTENTION_BACKEND.\n"
    "Question: How can you manually set the attention backend in vLLM?\n"
    "Answer: By configuring the environment "
    "variable VLLM_ATTENTION_BACKEND\n\n"

    "Context: vLLM exposes production metrics at the /metrics endpoint.\n"
    "Question: What endpoint does vLLM use to expose production metrics?\n"
    "Answer: /metrics\n\n"

    "Context: The three key abstractions used for disaggregated prefilling"
    " in vLLM are: KV pipe, KV lookup buffer, and KV connector.\n"
    "Question: What are the three key abstractions used for disaggregated"
    " prefilling in vLLM?\n"
    "Answer: KV pipe, KV lookup buffer, and KV connector\n\n"

    "Context: vLLM supports Expert Parallelism for large-scale deployment"
    " of Mixture of Experts models.\n"
    "Question: What parallelism strategy does vLLM support for large-scale"
    " deployment of Mixture of Experts models?\n"
    "Answer: Expert Parallelism\n\n"

    "Context: Each warp needs 8 inner iterations to handle a whole block"
    " of value tokens when BLOCK_SIZE=16, V_VEC_SIZE=8, HEAD_SIZE=128,"
    " WARP_SIZE=32.\n"
    "Question: How many inner iterations does a warp need to handle a whole"
    " block of value tokens when BLOCK_SIZE is 16, V_VEC_SIZE is 8,"
    " HEAD_SIZE is 128, and WARP_SIZE is 32?\n"
    "Answer: 8\n\n"

    "Context: MLP speculators are not compatible with pipeline parallelism.\n"
    "Question: What is the current limitation when using MLP speculators"
    " for speculative decoding in vLLM?\n"
    "Answer: Not compatible with pipeline parallelism\n\n"

    "Context: The three communication backends available for Expert"
    " Parallelism in vLLM are DeepEP, PPLX, and PyNCCL.\n"
    "Question: What are the three communication backends available for"
    " Expert Parallelism in vLLM?\n"
    "Answer: DeepEP, PPLX, and PyNCCL\n\n"

    "Context: vLLM cannot serve multiple models on a single port using"
    " the OpenAI API.\n"
    "Question: Can vLLM serve multiple models on a single port using"
    " the OpenAI API?\n"
    "Answer: No\n\n"

    "Context: The --reasoning-parser flag must be specified when serving"
    " a reasoning model in vLLM to extract reasoning content.\n"
    "Question: What flag do you need to specify when serving a reasoning"
    " model in vLLM to extract reasoning content?\n"
    "Answer: --reasoning-parser\n\n"

    "Context: To install vLLM for AWS Neuron run:"
    " pip install -U -r requirements/neuron.txt"
    " then VLLM_TARGET_DEVICE=neuron pip install -e .\n"
    "Question: How do you build and install vLLM from source for AWS Neuron?\n"
    "Answer: pip install -U -r requirements/neuron.txt then"
    " VLLM_TARGET_DEVICE=neuron pip install -e .\n\n"

    "Context: Disaggregated prefilling is used to tune time-to-first-token"
    " (TTFT) and inter-token latency (ITL) separately.\n"
    "Question: What are the two main reasons for using disaggregated"
    " prefilling in vLLM?\n"
    "Answer: Tuning TTFT and ITL separately\n\n"

    "Context: The minimum version of bitsandbytes required is 0.46.1.\n"
    "Question: What is the minimum version of bitsandbytes required for"
    " vLLM quantization?\n"
    "Answer: 0.46.1\n\n"

    "Context: Intel Gaudi requires Ubuntu 22.04 LTS, Python 3.10,"
    " an Intel Gaudi accelerator, and Intel Gaudi software version 1.18.0.\n"
    "Question: What are the system requirements for running vLLM with"
    " Intel Gaudi devices?\n"
    "Answer: Ubuntu 22.04 LTS, Python 3.10, Intel Gaudi accelerator,"
    " Intel Gaudi software version 1.18.0\n\n"

    "Context: vLLM troubleshooting for distributed deployments is documented"
    " at the distributed troubleshooting page.\n"
    "Question: Where can I find information about debugging distributed"
    " vLLM deployments?\n"
    "Answer: The distributed troubleshooting documentation page\n\n"
)

_RE_THINK = re.compile(r'<think>.*?</think>', re.DOTALL)
_RE_NEW_TURN = re.compile(
    r"(\nAnswer:.*|\nQuestion:.*|\nContext:.*|\nInstructions:.*)$",
    re.DOTALL | re.IGNORECASE
)
_RE_PREAMBLE = re.compile(
    r"^(Answer\s*:\s*|The answer is[:\s]+|Okay[,\s]+)",
    re.IGNORECASE
)
_RE_CODE_BLOCK = re.compile(r'```[\s\S]*?```|`[^`\n]+`')
_RE_MD_LINK = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
_RE_URL = re.compile(r'https?://\S+')
_RE_MD_HEADER = re.compile(r'^#{1,6}\s+', re.MULTILINE)


def clean_answer(text: str, question: str = "") -> str:
    text = _RE_THINK.sub('', text)
    text = _RE_NEW_TURN.sub('', text)
    text = _RE_PREAMBLE.sub('', text)
    if question.lower().startswith("where"):
        text = _RE_MD_LINK.sub(r'\1 (\2)', text)
    else:
        text = _RE_MD_LINK.sub(r'\1', text)
    text = _RE_CODE_BLOCK.sub('', text)
    if not question.lower().startswith("where"):
        text = _RE_URL.sub('', text)
    text = _RE_MD_HEADER.sub('', text)
    text = text.rstrip(' ,-')
    return " ".join(text.split()).strip()


def _make_llama_instance(
    model_path: str, n_ctx: int, n_threads: int, n_batch: int,
) -> Llama:
    with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            use_mlock=True,
            verbose=False,
            cache_prompt=True,
        )


class Reranker:
    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "in", "of", "to", "and",
        "for", "how", "what", "does", "do", "can", "you",
        "it", "with", "on", "are", "be", "by", "or", "at",
    })

    def _keyword_score(self, question: str, content: str) -> float:
        q_words = {
            w.lower() for w in re.findall(r'\w+', question)
            if w.lower() not in self._STOPWORDS and len(w) > 2
        }
        c_words = {
            w.lower() for w in re.findall(r'\w+', content)
            if w.lower() not in self._STOPWORDS and len(w) > 2
        }
        if not q_words:
            return 0.0
        return len(q_words & c_words) / len(q_words)

    def best_chunk(self, question: str, chunks: List[str]) -> str:
        if not chunks:
            return ""
        best = max(chunks, key=lambda c: self._keyword_score(question, c))
        truncated = best[:_MAX_CHUNK_CHARS]
        last_period = truncated.rfind(". ")
        if last_period > _MAX_CHUNK_CHARS // 2:
            truncated = truncated[:last_period + 1]
        return truncated


class SmallLLM:
    def __init__(self, n_workers: int = _BATCH_SIZE) -> None:
        hf_home = os.getenv("HF_HOME", "./.llm")
        model_path = os.path.join(hf_home, "hub", "Qwen")

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self._pool: List[Llama] = [
            _make_llama_instance(model_path, 2048, _THREADS_PER_WORKER, 256)
            for _ in range(n_workers)
        ]
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        print(f"[bold green]{n_workers} instances ready "
              f"({_THREADS_PER_WORKER} threads each).[/bold green]")

    def warmup(self, prefix: str) -> None:
        futures = {
            self._executor.submit(self._run_warmup, idx, prefix): idx
            for idx in range(len(self._pool))
        }
        for future in as_completed(futures):
            future.result()

    def _run_warmup(self, worker_idx: int, prefix: str) -> None:
        self._pool[worker_idx].create_completion(
            prompt=prefix,
            max_tokens=1,
            temperature=0.0,
        )

    def _run(self, worker_idx: int, prompt: str) -> Any:
        model = self._pool[worker_idx]
        outputs: Any = model.create_completion(
            prompt=prompt,
            max_tokens=_MAX_NEW_TOKENS,
            temperature=0.0,
            top_k=1,
            stop=["\n", "Question:", "Context:", "Instructions:", "```", "`"],
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

    def answer(self, question: str, context: str) -> None:
        prompt = self.build_prompt(question, context)
        raw = self.llm.generate_batch([prompt])[0]
        result = (
            clean_answer(raw, question) if raw else "Not found in context."
        )
        if not result:
            result = "Not found in context."
        print(f"[bold cyan]Question:[/bold cyan] {question}")
        print(f"[bold green]Answer:[/bold green] {result}")

    def _get_context(self, question: str, sources: list) -> str:
        chunks = [
            src.content.strip()
            for src in sources
            if src.content
            and src.content.strip()
            and len(src.content.strip()) > 50
            and not src.content.strip().startswith("# --8<--")
        ]
        return self.reranker.best_chunk(question, chunks)

    def build_prompt(self, question: str, context: str) -> str:
        return FEW_SHOT + f"Context: {context}\nQuestion: {question}\nAnswer:"

    def _prepare_batch(self, results_batch: list) -> List[str]:
        prompts = []
        for result in results_batch:
            ctx = self._get_context(
                result.question, result.retrieved_sources or []
            )
            prompts.append(
                self.build_prompt(result.question, ctx) if ctx else ""
            )
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

        print(f"[bold green]Processing {len(all_results)} questions.")

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
                final_answer = (
                    clean_answer(raw, result.question)
                    if raw else "Not found in context."
                )

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
                            "__all__": {"content"}
                        }
                    }
                ),
                fd,
                indent=4,
                ensure_ascii=False
            )
