from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """Abstract base class used to interface with (Language) Models.
    Current only one: Language.
    Likely more to be introduced if more types of LLMs are introduced (Speech, CNNs, Video...)
    """

    @abstractmethod
    def get_completion(self, prompt: str = "") -> str:
        pass
