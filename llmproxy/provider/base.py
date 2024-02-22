from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAdapter(ABC):
    """Abstract base class used to interface with (Language) Models.
    Current only one: Language.
    Likely more to be introduced if more types of LLMs are introduced (Speech, CNNs, Video...)
    """

    @abstractmethod
    def get_completion(self, prompt: str = "") -> str | None:
        """
        Abstract method to retrieve completion for a given prompt.

        Parameters:
        - prompt (str): The prompt for which completion is requested. Defaults to an empty string.

        Returns:
        - str or None: The completion for the provided prompt, or None if no completion is available.
        """

    @abstractmethod
    def get_estimated_max_cost(
        self, prompt: str = "", price_data: Dict[str, Any] = None
    ) -> float:
        """
        Abstract method to retrieve estimated for a given prompt.

        Parameters:
        - prompt (str): The prompt for which estimated cost is requested. Defaults to an empty string.

        Returns:
        - float: The estimated cost of the prompt
        """

    @abstractmethod
    def get_category_rank(self, category: str = "") -> int:
        """
        Abstract method to retrieve the rank of a given category.

        Parameters:
        - category (str): The category for which the rank is requested. Defaults to an empty string.

        Returns:
        - int: The rank of the specified category.
        """
