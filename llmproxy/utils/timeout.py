"""
Module for executing functions with timeouts using multiprocessing.

This module provides a function, timeout_function, which executes a given function
with a timeout using multiprocessing. It ensures that if the function takes longer
than the specified timeout, it returns None.

Example usage:
    res = timeout_function(some_function, 10)
    print(res)  # Prints the result of some_function if it completes within 10 seconds, else None.
"""
import multiprocessing
from typing import Any, Callable, Union


def worker(func: Callable, queue: multiprocessing.Queue, *args, **kwargs) -> None:
    """
    Worker function to execute the given function and put its result into a queue.

    Args:
        func: The function to execute.
        queue: The queue to put the result into.
    """
    try:
        result = func(*args, **kwargs)
        queue.put((result, None))
    except Exception as e:
        queue.put((None, e))


def timeout_function(
    func: Callable, timeout: int, *args, **kwargs
) -> Union[Any, Exception]:
    """
    Execute a function with a timeout and return its result.

    Args:
        func: The function to execute.
        timeout: The timeout value in seconds.

    Returns:
        The result of the function if it completes within the timeout.

    Raises:
        TimeoutError: If the function execution times out.
        Exception: If the function execution raises an exception.
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=worker, args=(func, queue, *args), kwargs=kwargs
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Function execution timed out.")

    result, error = queue.get()

    if error:
        raise error

    return result
