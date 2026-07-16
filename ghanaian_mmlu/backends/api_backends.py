"""
API-based model backends: OpenAI, Anthropic Claude, Google Gemini.
"""

import os
import re
import time
from typing import Optional
from .base import BaseModelBackend, ModelResponse


def _parse_label(text: str) -> Optional[str]:
    text = text.strip()
    m = re.match(r"^([A-H])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


class OpenAIBackend(BaseModelBackend):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None

    def load(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        print(f"OpenAI backend ready (model={self.model_name})")

    def predict(self, prompt: str) -> ModelResponse:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency = time.time() - t0

        raw_text = resp.choices[0].message.content or ""
        usage = resp.usage
        label = _parse_label(raw_text)

        return ModelResponse(
            raw_text=raw_text,
            predicted_label=label,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_s=latency,
        )

    def unload(self) -> None:
        self.client = None


class AnthropicBackend(BaseModelBackend):
    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None

    def load(self) -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
        print(f"Anthropic backend ready (model={self.model_name})")

    def predict(self, prompt: str) -> ModelResponse:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - t0

        raw_text = resp.content[0].text if resp.content else ""
        label = _parse_label(raw_text)

        return ModelResponse(
            raw_text=raw_text,
            predicted_label=label,
            prompt_tokens=resp.usage.input_tokens if resp.usage else None,
            completion_tokens=resp.usage.output_tokens if resp.usage else None,
            latency_s=latency,
        )

    def unload(self) -> None:
        self.client = None


class GeminiBackend(BaseModelBackend):
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None

    def load(self) -> None:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        print(f"Gemini backend ready (model={self.model_name})")

    def predict(self, prompt: str) -> ModelResponse:
        t0 = time.time()
        resp = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        latency = time.time() - t0

        raw_text = resp.text or ""
        usage = resp.usage_metadata
        label = _parse_label(raw_text)

        return ModelResponse(
            raw_text=raw_text,
            predicted_label=label,
            prompt_tokens=usage.prompt_token_count if usage else None,
            completion_tokens=usage.candidates_token_count if usage else None,
            latency_s=latency,
        )

    def unload(self) -> None:
        self.model = None
