class RequestsFailed(Exception):
    pass


class ModelRequestFailed(Exception):
    pass


class LLMProxyConfigError(Exception):
    pass


class UserConfigError(Exception):
    pass


class UserChatHistoryError(Exception):
    pass
