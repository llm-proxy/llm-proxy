import requests
from llmproxy.models.base import BaseChatbot, CompletionResponse
from llmproxy.utils.enums import BaseEnum


class Llama2Model(str, BaseEnum):
    LLAMA_2_7B = "Llama-2-7b-chat-hf"
    LLAMA_2_13B = "Llama-2-13b-chat-hf"
    LLAMA_2_70B = "Llama-2-70b-chat-hf"


class Llama2(BaseChatbot):
    def __init__(
        self,
        prompt: str = "",
        system_prompt: str = "Answer politely",
        api_key: str = "",
        temperature: float = 1.0,
        model: Llama2Model = Llama2Model.LLAMA_2_7B.value,
    ) -> None:
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.api_key = api_key
        self.temperature = temperature
        self.model = model

    def get_completion(self) -> CompletionResponse:
        if self.prompt == "":
            return self._handle_error(
                exception="No prompt detected", error_type="InputError"
            )
        if self.model not in Llama2Model:
            return self._handle_error(
                exception=f"Invalid Model. Please use one of the following model: {', '.join(Llama2Model.list_values())}",
                error_type="ValueError",
            )
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            API_URL = (
                f"https://api-inference.huggingface.co/models/meta-llama/{self.model}"
            )

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            # Llama2 prompt template
            prompt_template = f"<s>[INST] <<SYS>>\n{{{{ {self.system_prompt} }}}}\n<</SYS>>\n{{{{ {self.prompt} }}}}\n[/INST]"
            payload = {"inputs": prompt_template}
            output = query(payload)

        except Exception as e:
            raise Exception("Error Occur for prompting Llama2")

        if output["error"]:
            return self._handle_error(
                exception=output["error"], error_type="Llama2Error"
            )

        return CompletionResponse(payload=output, message="OK", err="")

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)