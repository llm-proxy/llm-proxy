from models.base import BaseChatbot, CompletionResponse
import requests

class Mistral:
    
    def __init__(
        self, 
        prompt: str = "", 
        api_key: str = "", 
        temp: float = 0,
    ) -> None:
        self.prompt = prompt
        self.api_key = api_key
        self.temp = temp

    def get_completion(self) -> CompletionResponse:

        try:
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()
                
            output = query({
                "inputs": self.prompt,
            })
        except Exception as e:
            raise Exception("Unknown Error")

        return CompletionResponse(
            payload=output[0]['generated_text'],
            message="OK",
            err="",
        )