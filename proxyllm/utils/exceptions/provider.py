class ModelException(Exception):
    """
    Exceptions for all supported models
    """


class CohereException(ModelException):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"Cohere Error: {exception}, Type: {error_type}")


class VertexAIException(ModelException):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"VertexAI Error: {exception}, Type: {error_type}")


class Llama2Exception(ModelException):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"Llama2 Error: {exception}, Type: {error_type}")


class MistralException(Exception):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"Mistral Error: {exception}, Type: {error_type}")


class OpenAIException(Exception):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"OpenAI Error: {exception}, Type: {error_type}")


class AnthropicException(Exception):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"Anthropic Error: {exception}, Type: {error_type}")


class UnsupportedModel(Exception):
    """
    General Exceptions shared across models
    """

    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"Unsupported Model Error: {exception}, Type: {error_type}")


class EmptyPrompt(Exception):
    pass
