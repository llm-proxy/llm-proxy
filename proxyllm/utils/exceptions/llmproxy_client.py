class RequestsFailed(Exception):
    pass


class ModelRequestFailed(Exception):
    pass


class LLMProxyConfigError(Exception):
    pass


class UserConfigError(Exception):
    pass


class UserChatHistoryError(Exception):
    def __init__(self, exception: str, error_type: str) -> None:
        super().__init__(f"Chat history Error: {exception}, Type: {error_type}")
