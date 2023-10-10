import requests
from models.base import BaseChatbot, CompletionResponse

class Llama2:

    def __init__(
        self,
        prompt: str = "",
        api_key: str = "",
        ) -> None:
        self.prompt = prompt
        self.api_key = api_key
    
    def get_completion(self) -> CompletionResponse:

        try:
            API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
            headers = {"Authorization": "Bearer {self.api_key}"}
            def query(payload):
                 response = requests.post(API_URL, headers=headers, json=payload)
                 return response.json()
            
            output = query({"inputs": self.prompt})

        except Exception as e:
            raise Exception("Error Occur")
        
        return CompletionResponse(
            payload=output
        )
