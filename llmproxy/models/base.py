from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionResponse:
    """
    payload: Data on successful response else ""
    message: Error message on unsuccessful response else "OK"
    err: Error type on unsuccessful response else ""
    """

    payload: str = ""
    message: str = ""
    err: str = ""


class BaseChatbot(ABC):
    @abstractmethod
    def get_completion(self, prompt: str = "", **kwargs: Any) -> CompletionResponse:
        pass
