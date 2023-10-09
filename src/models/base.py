from abc import ABC, abstractmethod


class BaseChatbot(ABC):
    @abstractmethod
    def get_completion(self, prompt, **Any):
        pass
