"""
Local model backend using HuggingFace transformers.
"""

import re
import time
import torch
from typing import Optional
from .base import BaseModelBackend, ModelResponse


class TransformersBackend(BaseModelBackend):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 1,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name} with transformers...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, "auto")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        elapsed = time.time() - t0
        print(f"Model loaded in {elapsed:.1f}s")

    def predict(self, prompt: str) -> ModelResponse:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.time() - t0

        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        prompt_toks = inputs["input_ids"].shape[-1]
        comp_toks = len(new_tokens)

        label = self._parse_label(raw_text)

        return ModelResponse(
            raw_text=raw_text,
            predicted_label=label,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            latency_s=latency,
        )

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
