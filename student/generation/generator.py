from student.models import (
    MinimalAnswer, MinimalSearchResults,
    StudentSearchResults, StudentSearchResultsAndAnswer,
)

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)

from student.utils import bar
from threading import Thread
from rich import print
from typing import Any

import torch as t
import json
import os


class SmallLLM:
    """Small local LLM wrapper optimised for fast inference."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        Load tokenizer and model on the best available device.

        Args:
            model_name: HuggingFace model identifier.
        """
        if t.cuda.is_available():
            self.device = t.device("cuda")
        else:
            self.device = t.device("cpu")

        self.dtype = (
            t.float16
            if self.device.type == "cuda"
            else t.float32
        )

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = (
            self.tokenizer.pad_token or self.tokenizer.eos_token
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        if self.device.type == "cuda":
            # 4-bit quantization drastically reduces VRAM usage
            # and improves inference throughput on consumer GPUs.
            #
            # This allows the model to:
            # - fit into smaller GPUs
            # - reduce memory bandwidth pressure
            # - generate responses significantly faster
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=t.float16
            )

            self.model: Any = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization_config,
                low_cpu_mem_usage=True, device_map="auto"
            )

            # Enable cuDNN autotuner for faster CUDA kernels.
            t.backends.cudnn.benchmark = True

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True,
            )

            self.model.to(self.device)

        self.model.eval()

        print(f"[bold green]Loaded {model_name} on {self.device}[/bold green]")


class Generator:
    """Generate answers using BM25 retrieval + local LLM."""

    def __init__(self) -> None:
        """Initialise the answer generator."""
        self.llm = SmallLLM()

    @staticmethod
    def batch_questions(
        questions: list[MinimalSearchResults]
    ) -> list[list[MinimalSearchResults]]:
        """
        Split questions into batches.

        Questions are sorted by length to reduce padding overhead
        during batched inference.

        Args:
            questions: List of search results.

        Returns:
            List of question batches.
        """
        batch_size = 16

        questions = sorted(questions, key=lambda q: len(q.question))

        return [
            questions[i:i + batch_size]
            for i in range(0, len(questions), batch_size)
        ]

    def build_prompt(self, result: MinimalSearchResults) -> str:
        """
        Build a chat prompt from a retrieved context.

        Args:
            result: Search result containing question and context.

        Returns:
            Formatted prompt string.
        """
        context = (result.content[:500] if result.content else "None")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Always respond in English. "
                    "Answer in 2 complete sentences "
                    "using only the context."
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

        return self.llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def answer_batch(
        self, batch: list[MinimalSearchResults], batch_idx: int,
    ) -> list[str]:
        """
        Generate answers for a batch of questions.

        Args:
            batch: Batch of retrieved search results.
            batch_idx: Batch index used for progress display.

        Returns:
            Generated answers.
        """
        tokenizer = self.llm.tokenizer

        prompts = [self.build_prompt(result) for result in batch]

        inputs = tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True,
            max_length=512, pad_to_multiple_of=8,
        ).to(self.llm.device)

        outputs: Any = None
        error: Exception | None = None

        def run_model() -> None:
            """Run generation in a background thread."""
            nonlocal outputs
            nonlocal error

            try:
                with t.inference_mode():
                    outputs = self.llm.model.generate(
                        **inputs, max_new_tokens=100,
                        do_sample=False, use_cache=True,
                        num_beams=1, repetition_penalty=1.3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except Exception as exc:
                error = exc

        thread = Thread(target=run_model)
        thread.start()

        with bar(
            total=None, desc=f"Batch {batch_idx}",
            position=1, leave=False,
            color="red"
        ) as pbar:
            while thread.is_alive():
                thread.join(timeout=0.05)
                pbar.update(1)

        thread.join()

        if error is not None:
            raise error

        if outputs is None:
            raise RuntimeError("Generation failed.")

        answers: list[str] = []

        for i, output in enumerate(outputs):
            # Remove the original prompt tokens from the generated output.
            #
            # The model returns:
            # [prompt tokens] + [generated tokens]
            #
            # We only want the generated answer.
            prompt_len = inputs["input_ids"][i].shape[0]

            generated_tokens = output[prompt_len:]

            answer = tokenizer.decode(
                generated_tokens, skip_special_tokens=True,
            ).strip()

            # Remove possible leftover assistant tags or thinking tags.
            answer = answer.replace("<think>", "")
            answer = answer.replace("</think>", "")
            answer = answer.replace("assistant", "")

            # Normalise whitespace.
            answer = " ".join(answer.split())

            answers.append(answer)

        return answers

    def answer(self, question: str, context: list[str] | str) -> None:
        """
        Answer a single question and save the result.

        Args:
            question: User question.
            context: Retrieved context passages.
        """
        if isinstance(context, list):
            context = "\n".join(context)

        result = MinimalSearchResults(
            question_id="single", question=question,
            retrieved_sources=[], content=context
        )

        answer_text = self.answer_batch([result], 1)[0]

        output = {
            "question": question, "answer": answer_text, "context": context,
        }

        os.makedirs("data/output", exist_ok=True)

        output_path = "data/output/single_answer.json"

        with open(output_path, "w", encoding="utf-8") as fd:
            json.dump(output, fd, indent=4, ensure_ascii=False)

        print(f"[bold green]Saved answer to:\n{output_path}[/bold green]")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str,
    ) -> None:
        """
        Generate answers for a full dataset.

        Args:
            student_search_results_path:
                Path to search results JSON.
            save_directory:
                Directory where outputs will be saved.
        """
        with open(student_search_results_path, "r", encoding="utf-8") as fd:
            raw = json.load(fd)

        results = StudentSearchResults.model_validate(raw)

        total = len(results.search_results)

        batches = self.batch_questions(results.search_results)

        os.system("clear")

        print(f"[bold green]Loaded {total} questions[/bold green]")

        answers: list[MinimalAnswer] = []

        for batch_idx, batch in enumerate(bar(
            data=batches, desc="Generating answers",
            position=0, leave=True, color="green"), start=1
        ):
            batch_answers = self.answer_batch(batch, batch_idx)

            for result, answer_text in zip(batch, batch_answers):
                answers.append(MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text
                ))

        print(f"[bold green]Processed {total}/{total} questions[/bold green]")

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k,
        )

        os.makedirs(save_directory, exist_ok=True)

        file_name = os.path.basename(student_search_results_path)

        output_path = os.path.join(save_directory, file_name)

        with open(output_path, "w", encoding="utf-8") as fd:
            json.dump(
                output.model_dump(
                    exclude={
                        "search_results": {
                            "__all__": {"content"}
                            }
                        }
                ),
                fd, indent=4, ensure_ascii=False,
            )

        print(f"\n[bold green]Saved results to:\n{output_path}[/bold green]")
