import requests
from llmproxy.models.base import BaseModel, CompletionResponse
from llmproxy.utils.enums import BaseEnum
from llmproxy.utils.log import logger
from llmproxy.utils import tokenizer

# This is not accurate data
llama2_price_data = {
    "max-output-tokens": 50,
    "model-costs": {
        "prompt": 1.10 / 1_000_000,
        "completion": 1.80 / 1_000_000,
    },
}


class Llama2Model(str, BaseEnum):
    LLAMA_2_7B = "Llama-2-7b-chat-hf"
    LLAMA_2_13B = "Llama-2-13b-chat-hf"
    LLAMA_2_70B = "Llama-2-70b-chat-hf"


class Llama2(BaseModel):
    def __init__(
        self,
        prompt: str = "",
        system_prompt: str = "Answer politely",
        api_key: str = "",
        temperature: float = 1.0,
        model: Llama2Model = Llama2Model.LLAMA_2_7B.value,
        max_output_tokens: int = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.api_key = api_key
        self.temperature = temperature
        self.model = model
        self.max_output_tokens = max_output_tokens

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
                response = requests.post(
                    API_URL, headers=headers, json=payload)
                return response.json()

            # Llama2 prompt template
            prompt_template = f"<s>[INST] <<SYS>>\n{{{{ {self.system_prompt} }}}}\n<</SYS>>\n{{{{ {self.prompt} }}}}\n[/INST]"
            output = query(
                {
                    "inputs": prompt_template,
                    "parameters": {
                        "max_length": self.max_output_tokens,
                        "temperature": self.temperature,
                    },
                }
            )

        except Exception as e:
            raise Exception("Error Occur for prompting Llama2")

        if output["error"]:
            return self._handle_error(
                exception=output["error"], error_type="Llama2Error"
            )

        return CompletionResponse(payload=output, message="OK", err="")

    def get_estimated_max_cost(self, prompt: str = "") -> float:
        if not self.prompt and not prompt:
            logger.info("No prompt provided.")
            raise ValueError("No prompt provided.")

        # Assumption, model exists (check should be done at yml load level)
        logger.info(f"Tokenizing model: {self.model}")

        prompt_cost_per_token = llama2_price_data["model-costs"]["prompt"]
        logger.info(f"Prompt Cost per token: {prompt_cost_per_token}")

        completion_cost_per_token = llama2_price_data["model-costs"]["completion"]
        logger.info(f"Output cost per token: {completion_cost_per_token}")

        tokens = tokenizer.bpe_tokenize_encode(
            prompt if prompt else self.prompt)

        logger.info(f"Number of input tokens found: {len(tokens)}")

        logger.info(
            f"Final calculation using {len(tokens)} input tokens and {llama2_price_data['max-output-tokens']} output tokens"
        )

        cost = round(
            prompt_cost_per_token * len(tokens)
            + completion_cost_per_token
            * llama2_price_data["max-output-tokens"],
            8,
        )

        logger.info(f"Calculated Cost: {cost}")

        return cost

    def _handle_error(self, exception: str, error_type: str) -> CompletionResponse:
        return CompletionResponse(message=exception, err=error_type)
