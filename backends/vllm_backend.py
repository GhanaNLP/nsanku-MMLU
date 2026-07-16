"""
Local model backend using vLLM (high-throughput inference).
"""

import re
import time
from typing import Optional
from .base import BaseModelBackend, ModelResponse


class VLLMBackend(BaseModelBackend):
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.sampling_params = None

    def load(self) -> None:
        from vllm import LLM, SamplingParams

        print(f"Loading {self.model_name} with vLLM...")
        t0 = time.time()

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1.0 if self.temperature > 0 else None,
        )

        elapsed = time.time() - t0
        print(f"Model loaded in {elapsed:.1f}s")

    def predict(self, prompt: str) -> ModelResponse:
        t0 = time.time()
        outputs = self.llm.generate([prompt], self.sampling_params)
        latency = time.time() - t0

        raw_text = outputs[0].outputs[0].text
        prompt_toks = len(outputs[0].prompt_token_ids)
        comp_toks = len(outputs[0].outputs[0].token_ids)

        label = self._parse_label(raw_text)

        return ModelResponse(
            raw_text=raw_text,
            predicted_label=label,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            latency_s=latency,
        )

    def predict_batch(self, prompts: list[str]) -> list[ModelResponse]:
        """Batch inference — main advantage of vLLM."""
        t0 = time.time()
        outputs = self.llm.generate(prompts, self.sampling_params)
        total_latency = time.time() - t0

        results = []
        for out in outputs:
            raw_text = out.outputs[0].text
            prompt_toks = len(out.prompt_token_ids)
            comp_toks = len(out.outputs[0].token_ids)
            label = self._parse_label(raw_text)
            results.append(
                ModelResponse(
                    raw_text=raw_text,
                    predicted_label=label,
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    latency_s=total_latency / len(prompts),
                )
            )
        return results

    def unload(self) -> None:
        self.llm = None

    @staticmethod
    def _parse_label(text: str) -> Optional[str]:
        text = text.strip()
        m = re.match(r"^([A-H])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-H])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None
