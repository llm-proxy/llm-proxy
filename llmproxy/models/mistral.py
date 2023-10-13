from llmproxy.models.base import BaseChatbot, CompletionResponse
import requests


class Mistral:
    INPUT_COST_PER_TOKEN = 0
    OUTPUT_COST_PER_TOKEN = 50
    SPECIALIZATIONS = set(("MATH", "SCIENCE"))

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
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            output = query(
                {
                    "inputs": self.prompt,
                }
            )
        except Exception as e:
            raise Exception(e)

        response = ""
        if isinstance(output, list) and "generated_text" in output[0]:
            response = output[0]["generated_text"]
        elif "error" in output:
            response = "ERROR: " + output["error"]
        else:
            raise ValueError("Unknown output format")

        return CompletionResponse(
            payload=response,
            message="OK",
            err="",
        )
