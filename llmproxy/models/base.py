from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

class BaseModel(ABC):
    """Abstract based class used to interface with (Language) Models.
    Current only one: Language.
    Likely more to be introduced if more types of LLMs are introduced (Speech, CNNs, Video...)
    """

    @abstractmethod
    def get_completion(self, **kwargs: Any) -> str:
        pass
