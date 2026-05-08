from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch as t


class SmallLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B"
    ) -> None:

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

        self.model: PreTrainedModel = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> t.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return t.tensor([ids], device=self.device)


class Generator:
    def __init__(
        self, model_name: str = "Qwen/Qwen3-0.6B",
        device: str | None = None,
        dtype: t.dtype | None = None
    ) -> None:
        self._model_name = model_name
        self._device = device or (
            "mps" if t.backends.mps.is_available()
            else "cuda" if t.cuda.is_available()
            else "cpu"
        )
        self._dtype = (
            dtype
            if dtype is not None
            else t.float16 if self._device in ["cuda", "mps"]
            else t.float32
        )

    def _encode(self, text: str) -> t.Tensor:
        pass

    def answer(prompt, sources) -> None:
        pass

    def answer_dataset(student_search_results_path, save_directory) -> None:
        pass
