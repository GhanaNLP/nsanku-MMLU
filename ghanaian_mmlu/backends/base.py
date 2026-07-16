"""
Base interface for all model backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelResponse:
    raw_text: str
    predicted_label: Optional[str]  # A/B/C/D or None if parse failed
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_s: Optional[float] = None


class BaseModelBackend(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, prompt: str) -> ModelResponse:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()
