from student.models import (
    MinimalSource, MinimalAnswer, StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Any
import torch as t
import json
import os


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
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()


class Generator:
    def __init__(self) -> None:
        self.llm = SmallLLM()

    def answer(self, prompt: str, sources: list[MinimalSource]) -> str:
        context = "\n\n".join([
            f"Source ({s.file_path} "
            f"[{s.first_character_index}:{s.last_character_index}])"
            for s in sources
        ])
        full_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer based only on the context above:"
        )
        tokenizer = self.llm.tokenizer
        model = self.llm.model
        inputs = tokenizer(
            full_prompt, return_tensors="pt"
        ).to(self.llm.device)
        with t.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        with open(student_search_results_path, "r") as fd:
            raw = json.load(fd)

        results = StudentSearchResults.model_validate(raw)
        answers = []

        for result in tqdm(results.search_results, desc="Generating answers"):
            answer_text = self.answer(
                result.question, result.retrieved_sources
            )
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=results.k
        )

        os.makedirs(save_directory, exist_ok=True)
        file_name = student_search_results_path.rsplit("/", 1)[1]
        with open(f"{save_directory}/{file_name}", "w") as fd:
            json.dump(output.model_dump(), fd, indent=4)
