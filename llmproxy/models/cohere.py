import cohere
from llmproxy.models.base import BaseChatbot, CompletionResponse
from llmproxy.utils.enums import BaseEnum

class CohereModel(str, BaseEnum):
    COMMAND = "command"
    COMMAND_LIGHT = "command-light"
    COMMAND_NIGHTLY = "command-nightly"
    COMMAND_LIGHT_NIGHTLY = "command-light-nightly"

class Cohere(BaseChatbot):
    def __init__(
        self,
        message: str = "",
        model: CohereModel = CohereModel.COMMAND,
        temperature: float = 0,
        api_key: str = "",
    ) -> None:
        self.message = message
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.co = None
        try: 
            self.co = cohere.Client(self.api_key)
        except cohere.CohereError as e:
            self.error_response = CompletionResponse(
               payload="",
               message=e,
               err="ValueError"
           )  

    def get_completion(self) -> CompletionResponse:
        if self.co is None:
           return self.error_response
        try:
            response = self.co.chat(
            message = self.message,
            model = " ",
            connectors = [{"id": "web-search"}], # perform web search before answering the question
            temperature = self.temperature            
            )
            return CompletionResponse(
                payload=response.text,
                message="OK",
                err="",
            )
        except cohere.CohereError as e:
            return CompletionResponse(
                payload="",
                message=e.message,
                err=e.__class__,
            )
        except Exception as e:
            raise Exception("unknown error")
    