import time

import pytest

from llmproxy.utils import timeout


def sleep_function(seconds: int):
    time.sleep(seconds)


def test_timeout_wrapper_completion_within_timeout():
    assert timeout.timeout_wrapper(sleep_function, 3, 0) is None


def test_timeout_wrapper_raises_timeout_error():
    with pytest.raises(TimeoutError):
        timeout.timeout_wrapper(sleep_function, 0, 10)
