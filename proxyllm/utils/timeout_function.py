import multiprocessing
import threading
from typing import Any, Callable, Union


def timeout_wrapper(func: Callable, timeout: int, *args, **kwargs):
    """
    Wrap a function call in a thread with a timeout.

    Parameters:
        func (callable): The function to be called.
        args (tuple): The arguments to pass to the function.
        timeout (int): The maximum time to wait for the function to complete, in seconds.

    Raises:
        TimeoutError: If the function does not complete within the specified timeout.

    Note:
        This function creates a daemon thread to execute the given function with the provided arguments.
        If the function does not complete within the specified timeout, a TimeoutError is raised.
    """

    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    # make thread daemon so it can run in background
    # Allows main thread to exit without child thread
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("Operation timed out")


"""
Module for executing functions with timeouts using multiprocessing and threading.

This module provides two functions:
1. timeout_function: Executes a given function with a timeout using multiprocessing. It ensures that if the function takes longer than the specified timeout, it returns None.
2. timeout_wrapper: Wraps a function call in a thread with a timeout using threading. If the function does not complete within the specified timeout, it raises a TimeoutError.

Example usage:
    res = timeout_function(some_function, 10)
    print(res)  # Prints the result of some_function if it completes within 10 seconds, else None.

    try:
        timeout_wrapper(some_function, 5)
    except TimeoutError as e:
        print(e) # Prints "Operation timed out" if some_function does not complete within 5 seconds.
"""


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
