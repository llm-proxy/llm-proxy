import time

import pytest

from proxyllm.utils import timeout_function


def sleep_function(seconds: int):
    time.sleep(seconds)


def test_timeout_function_wrapper_completion_within_timeout():
    assert timeout_function.timeout_wrapper(sleep_function, 3, 0) is None


def test_timeout_function_wrapper_raises_timeout_error():
    with pytest.raises(TimeoutError):
        timeout_function.timeout_wrapper(sleep_function, 0, 10)
