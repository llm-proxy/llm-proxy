from llmproxy.models.base import BaseChatbot, CompletionResponse
from llmproxy.utils.enums import BaseEnum
import requests

class MistralModel(str, BaseEnum):
    Mistral_7B = "Mistral-7B-v0.1"
    Mistral_7B_Instruct = "Mistral-7B-Instruct-v0.1"
class Mistral(BaseChatbot):
    def __init__(
        self, 
        prompt: str = "", 
        model: MistralModel = MistralModel.Mistral_7B_Instruct.value,
        api_key: str = "", 
        temp: float = 1,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.temp = temp

    def get_completion(self) -> CompletionResponse:
        if self.model not in MistralModel:
            models = [model.value for model in MistralModel]
            return CompletionResponse(message="Model not supported, please use one of the following:\n" + "\n".join(models), err="ValueError")
        try:
            API_URL = f"https://api-inference.huggingface.co/models/mistralai/{self.model}"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()
                
            output = query({
                "inputs": self.prompt,
                "parameters": {"temperature": self.temp}
            })
        except Exception as e:
            raise Exception(e)
        
        response = ""
        message = ""
        if isinstance(output, list) and 'generated_text' in output[0]:
            response = output[0]['generated_text']
        elif 'error' in output:
            message = "ERROR: " + output['error']
        else:
            raise ValueError("Unknown output format")

        return CompletionResponse(
            payload=response,
            message=message,
            err="",
        )