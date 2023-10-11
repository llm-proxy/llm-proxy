from enum import Enum, EnumMeta
from llmproxy.models.openai import OpenAI
class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass

#testing default model
def test_open_ai_default() -> None:
    openai = OpenAI()
    assert(openai.model == "gpt-3.5-turbo")