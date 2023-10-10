import cohere
from models.base import BaseChatbot, CompletionResponse
from utils.enums import BaseEnum


@dataclass
class CompletionResponse:
    payload: str
    message: str
    err: str

class CohereModel(str, BaseEnum)
    COMMAND = "command"
    COMMAND_LIGHT = "command-light"
    COMMAND_NIGHTLY = "command-nightly"
    COMMAND_LIGHT_NIGHTLY = "command-light-nightly"


class Cohere(BaseChatBot)
    def __init__(
        self,
        message: str = "",
        model: CohereModel = CohereModel.COMMAND,
        temperature: float = 0,
        api_key: str = "",
    ) -> None:
        self.message = message,
        self.model = model,
        self.temperature = temperature,
        Cohere.api_key = api_key
    
    def cohere_ai_completion(self) -> CompletionResponse:
        try:
            co = cohere.Client(Cohere.api_key)
            response = co.chat(
            message=prompt,
            connectors=[{"id": "web-search"}] # perform web search before answering the question
            )
        except cohere.CohereError as e:
            return CompletionResponse(
                payload="",
                message="FAILED",
                err=e.message,
            )
        return CompletionResponse(
            payload=response.text,
            message="OK",
            err="",
        )
    