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
        prompt: str = "",
        model: CohereModel = CohereModel.COMMAND,
        temperature: float = 0,
        api_key: str = "",
        max_token: int = 0,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_token = max_token
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
        if self.model not in CohereModel:
            return CompletionResponse(
                payload="",
                message=f"Model not supported. Please use one of the following models: {', '.join(Cohere.list_values())}",
                err="ValueError",
            )
        if self.co is None:
           return self.error_response
        try:
            response = self.co.chat(
                max_tokens = self.max_token, 
                message = self.prompt,
                model = self.model,
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
                err=e.http_status
            )
        except Exception as e:
            raise Exception("Unknown Cohere error when making API call")