"""
API-based model backends: OpenAI, Anthropic Claude, Google Gemini, NVIDIA Build.
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
        disable_thinking: bool = True,
    ):
        # Uses the current google.genai SDK (the legacy google.generativeai
        # package is deprecated and cannot control thinking budget).
        self.model_name = model_name
        # Client() also auto-reads GEMINI_API_KEY / GOOGLE_API_KEY from the env.
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Gemini 3 is a thinking model; disable thinking (thinking_budget=0) so
        # it is compared on equal footing with the other non-thinking models.
        # NOTE: gemini-3.x-*-pro models reject budget=0 ("only works in thinking
        # mode") and cannot be run in this standardized no-thinking benchmark.
        self.disable_thinking = disable_thinking
        self.client = None

    def load(self) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=self.api_key)
        print(
            f"Gemini backend ready (model={self.model_name}, "
            f"disable_thinking={self.disable_thinking})"
        )

    def predict(self, prompt: str) -> ModelResponse:
        types = self._types
        t0 = time.time()

        cfg = dict(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.disable_thinking:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg),
        )
        latency = time.time() - t0

        # Gemini raises when a candidate has no valid Part (e.g. safety block or
        # empty finish_reason). Treat those as an empty/unparseable response
        # instead of crashing the whole evaluation run.
        try:
            raw_text = resp.text or ""
        except (ValueError, IndexError, AttributeError):
            raw_text = ""
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
        self.client = None


class NVIDIABuildBackend(BaseModelBackend):
    """NVIDIA Build API (OpenAI-compatible endpoint)."""

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-v4-flash",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        disable_thinking: bool = True,
        min_request_interval: float = 1.5,
    ):
        self.model_name = model_name
        # Support a POOL of keys so we can rotate past a key that has hit its
        # daily/rate quota (429). Sources, in priority order:
        #   1. explicit api_key argument
        #   2. NVIDIA_BUILD_API_KEYS  (comma-separated pool)
        #   3. NVIDIA_BUILD_API_KEY   (single key)
        raw = (
            api_key
            or os.environ.get("NVIDIA_BUILD_API_KEYS")
            or os.environ.get("NVIDIA_BUILD_API_KEY")
            or ""
        )
        self.api_keys = [k.strip() for k in raw.split(",") if k.strip()]
        self._key_idx = 0
        self.max_tokens = max_tokens
        self.temperature = temperature
        # These are reasoning models; disable thinking so all models are
        # compared on equal footing (no chain-of-thought).
        self.disable_thinking = disable_thinking
        # The NVIDIA Build free tier rate-limits (429). Pace requests to stay
        # under the limit instead of bursting into it. Overridable via the env
        # var NVIDIA_MIN_INTERVAL (seconds between requests).
        self.min_request_interval = float(
            os.environ.get("NVIDIA_MIN_INTERVAL", min_request_interval)
        )
        self._last_request_t = 0.0
        self.client = None

    def load(self) -> None:
        from openai import OpenAI
        self._OpenAI = OpenAI
        self.client = OpenAI(api_key=self.api_keys[self._key_idx], base_url=self.BASE_URL)
        print(
            f"NVIDIA Build backend ready (model={self.model_name}, "
            f"{len(self.api_keys)} key(s) in pool)"
        )

    def _rotate_key(self) -> int:
        """Switch to the next key in the pool (wraps around)."""
        self._key_idx = (self._key_idx + 1) % len(self.api_keys)
        self.client = self._OpenAI(
            api_key=self.api_keys[self._key_idx], base_url=self.BASE_URL
        )
        return self._key_idx

    MAX_RETRIES = 40

    def predict(self, prompt: str) -> ModelResponse:
        import openai as _openai
        extra = {"reasoning_effort": "none"} if self.disable_thinking else {}

        # Throttle: keep a minimum gap between successive requests.
        gap = time.time() - self._last_request_t
        if gap < self.min_request_interval:
            time.sleep(self.min_request_interval - gap)

        t0 = time.time()
        resp = None
        n_keys = len(self.api_keys)
        retryable = (
            _openai.InternalServerError,
            _openai.RateLimitError,
            _openai.APITimeoutError,
            _openai.APIConnectionError,
        )
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **extra,
                )
                self._last_request_t = time.time()
                break
            except _openai.BadRequestError as e:
                # Non-thinking models reject reasoning_effort. Drop it once and
                # retry (and stop sending it for the rest of the run).
                if extra:
                    print("\n  [model rejected reasoning_effort; retrying "
                          "without it (this model has no thinking to disable)]",
                          flush=True)
                    extra = {}
                    continue
                raise
            except retryable as e:
                if attempt >= self.MAX_RETRIES - 1:
                    raise
                is_429 = isinstance(e, _openai.RateLimitError)
                if is_429 and n_keys > 1:
                    # A key is throttled -> immediately try the next key.
                    idx = self._rotate_key()
                    # Only pause once we've cycled through the whole pool and
                    # every key was throttled (real sustained limit).
                    if (attempt + 1) % n_keys == 0:
                        wait = min(2 ** (attempt // n_keys) * 5, 60)
                        print(f"\n  [all {n_keys} keys throttled, waiting {wait}s...]",
                              flush=True)
                        time.sleep(wait)
                    else:
                        print(f"\r  [429 -> rotating to key #{idx+1}/{n_keys}]  ",
                              end="", flush=True)
                else:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"\n  [retry {attempt+1}/{self.MAX_RETRIES} "
                          f"after {type(e).__name__}, waiting {wait}s...]", flush=True)
                    time.sleep(wait)
        latency = time.time() - t0

        # Some models/endpoints occasionally return an empty `choices` list or a
        # null message; treat that as an unparseable response rather than
        # crashing the whole run (mirrors the Gemini backend).
        try:
            raw_text = resp.choices[0].message.content or ""
        except (IndexError, AttributeError):
            raw_text = ""
        usage = resp.usage if resp is not None else None
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


class MistralBackend(BaseModelBackend):
    """Mistral API (OpenAI-compatible endpoint)."""

    BASE_URL = "https://api.mistral.ai/v1"
    MAX_RETRIES = 8

    def __init__(
        self,
        model_name: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        min_request_interval: float = 1.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_request_interval = float(
            os.environ.get("MISTRAL_MIN_INTERVAL", min_request_interval)
        )
        self._last_request_t = 0.0
        self.client = None

    def load(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        print(f"Mistral backend ready (model={self.model_name})")

    def predict(self, prompt: str) -> ModelResponse:
        import openai as _openai

        gap = time.time() - self._last_request_t
        if gap < self.min_request_interval:
            time.sleep(self.min_request_interval - gap)

        t0 = time.time()
        resp = None
        retryable = (
            _openai.InternalServerError,
            _openai.RateLimitError,
            _openai.APITimeoutError,
            _openai.APIConnectionError,
        )
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                self._last_request_t = time.time()
                break
            except retryable as e:
                if attempt >= self.MAX_RETRIES - 1:
                    raise
                wait = min(2 ** attempt * 5, 60)
                print(f"\n  [retry {attempt+1}/{self.MAX_RETRIES} "
                      f"after {type(e).__name__}, waiting {wait}s...]", flush=True)
                time.sleep(wait)
        latency = time.time() - t0

        try:
            raw_text = resp.choices[0].message.content or ""
        except (IndexError, AttributeError):
            raw_text = ""
        usage = resp.usage if resp is not None else None
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
