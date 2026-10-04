from copy import deepcopy
from typing import Any, Dict, List

from proxyllm.provider.base import BaseAdapter, TokenizeResponse
from proxyllm.utils import proxy_logger
from proxyllm.utils.exceptions.provider import DeepSeekException


class DeepSeekAdapter(BaseAdapter):
    """Adapter for interacting with DeepSeek's language models."""

    def __init__(
        self,
        prompt: str = "",
        model: str = "",
        temperature: float = 0,
        api_key: str = "",
        max_output_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_output_tokens = (
            256 if max_output_tokens is None else max_output_tokens
        )
        self.timeout = 60 if timeout is None else timeout

    def get_completion(
        self,
        prompt: str = "",
        chat_history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any] | None:
        """
        Request a completion from the selected DeepSeek model.

        Returns:
            The response text and updated chat history.

        Raises:
            DeepSeekException: If authentication or the request fails.
        """
        if not self.api_key:
            raise DeepSeekException(
                exception="EMPTY API KEY: API key not provided",
                error_type="No API Key Provided",
            )

        text = prompt or self.prompt

        if not text.strip():
            raise DeepSeekException(
                exception="EMPTY PROMPT: Prompt not provided",
                error_type="EmptyPrompt",
            )

        from openai import OpenAI, OpenAIError

        try:
            messages = deepcopy(chat_history) if chat_history is not None else []
            messages.append({"role": "user", "content": text})

            with OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                timeout=self.timeout,
                max_retries=0,
            ) as client:
                response = client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    extra_body={"thinking": {"type": "disabled"}},
                )

            response_text = response.choices[0].message.content

            if not response_text:
                raise DeepSeekException(
                    exception="Model returned no text",
                    error_type="EmptyResponse",
                )

            messages.append(
                {"role": "assistant", "content": response_text}
            )

            return {
                "response": response_text,
                "chat_history": messages,
            }

        except DeepSeekException:
            raise
        except OpenAIError as error:
            raise DeepSeekException(
                exception=str(error),
                error_type=type(error).__name__,
            ) from error
        except Exception as error:
            raise DeepSeekException(
                exception=str(error),
                error_type="Unknown DeepSeek Error",
            ) from error

    def tokenize(self, prompt: str = "") -> TokenizeResponse:
        """
        Estimate input tokens and return the configured output limit.

        UTF-8 byte length is a conservative text estimate, not an exact
        DeepSeek token count. It excludes chat history and message framing.
        """
        text = prompt or self.prompt

        return TokenizeResponse(
            num_of_input_tokens=len(text.encode("utf-8")),
            num_of_output_tokens=self.max_output_tokens,
        )

    def get_category_rank(self, category: str = "") -> float:
        """
        Return an unmeasured rank until category benchmarks are available.

        Models with measured ranks are attempted first.
        """
        proxy_logger.log(msg=f"MODEL: {self.model}", color="PURPLE")

        category_rank = float("inf")

        proxy_logger.log(
            msg=f"MODEL CATEGORY RANK: {category_rank}",
            color="BLUE",
        )

        return category_rank