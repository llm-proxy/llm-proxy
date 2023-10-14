import requests
from llmproxy.models.base import BaseChatbot, CompletionResponse


class Llama2(BaseChatbot):
    INPUT_COST_PER_TOKEN = 0
    OUTPUT_COST_PER_TOKEN = 45
    SPECIALIZATIONS = set(("SCIENCE"))

    def __init__(
        self,
        prompt: str = "",
        system_prompt: str = "",
        api_key: str = "",
    ) -> None:
        self.system_prompt = system_prompt or "Answer politely"
        self.prompt = prompt
        self.API_URL = (
            "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        )
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_completion(self) -> CompletionResponse:
        if self.prompt == "":
            return self._handle_error(
                exception="No prompt detected", error_type="InputError"
            )
        try:

            def query(payload):
                response = requests.post(
                    self.API_URL, headers=self.headers, json=payload
                )
                return response.json()

            # Llama2 prompt template
            prompt_template = f"<s>[INST] <<SYS>>\n{{{{ {self.system_prompt} }}}}\n<</SYS>>\n{{{{ {self.prompt} }}}}\n[/INST]"
            payload = {"inputs": prompt_template}
            output = query(payload)

        except Exception as e:
            raise Exception("Error Occur")

        if output["error"]:
            return self._handle_error(exception=output["error"], error_type="..")

        return CompletionResponse(payload=output, message="OK", err="")

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
