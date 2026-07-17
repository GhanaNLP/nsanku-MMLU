"""
Factory to create the appropriate backend based on model specification.
"""

from typing import Optional
from .base import BaseModelBackend


def create_backend(
    model_spec: str,
    *,
    api_key: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    **kwargs,
) -> BaseModelBackend:
    """
    Create a backend from a model specification string.

    Prefix convention:
        openai/    -> OpenAI API (e.g. openai/gpt-4o)
        anthropic/ -> Anthropic API (e.g. anthropic/claude-sonnet-4-20250514)
        gemini/    -> Google Gemini (e.g. gemini/gemini-2.5-flash)
        nvidia/    -> NVIDIA Build API (e.g. nvidia/deepseek-ai/deepseek-v4-flash)
        vllm/      -> vLLM local (e.g. vllm/meta-llama/Llama-3-8B)
        (none)     -> transformers local (e.g. meta-llama/Llama-3-8B)
    """
    if model_spec.startswith("openai/"):
        from .api_backends import OpenAIBackend

        return OpenAIBackend(
            model_name=model_spec[len("openai/"):],
            api_key=api_key,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif model_spec.startswith("anthropic/"):
        from .api_backends import AnthropicBackend

        return AnthropicBackend(
            model_name=model_spec[len("anthropic/"):],
            api_key=api_key,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif model_spec.startswith("gemini/"):
        from .api_backends import GeminiBackend

        return GeminiBackend(
            model_name=model_spec[len("gemini/"):],
            api_key=api_key,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif model_spec.startswith("nvidia/"):
        from .api_backends import NVIDIABuildBackend

        return NVIDIABuildBackend(
            model_name=model_spec[len("nvidia/"):],
            api_key=api_key,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif model_spec.startswith("mistral/"):
        from .api_backends import MistralBackend

        return MistralBackend(
            model_name=model_spec[len("mistral/"):],
            api_key=api_key,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif model_spec.startswith("vllm/"):
        from .vllm_backend import VLLMBackend

        return VLLMBackend(
            model_name=model_spec[len("vllm/"):],
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
    else:
        from .transformers_backend import TransformersBackend

        return TransformersBackend(
            model_name=model_spec,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
