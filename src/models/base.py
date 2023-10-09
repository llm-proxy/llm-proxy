from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResponse:
    payload: str = ""
    message: str = ""
    err: str = ""


class BaseChatbot(ABC):
    @abstractmethod
    def get_completion(self, prompt, **Any) -> CompletionResponse:
        pass
