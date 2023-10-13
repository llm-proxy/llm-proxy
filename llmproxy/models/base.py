from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionResponse:
    payload: str = ""
    message: str = ""
    err: str = ""


class BaseChatbot(ABC):
    @abstractmethod
    def get_completion(self, prompt: str = "", **kwargs: Any) -> CompletionResponse:
        pass
