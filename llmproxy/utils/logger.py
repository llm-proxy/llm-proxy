import datetime as dt
import logging
import sys
from typing import Literal


class CustomFormatter(logging.Formatter):
    """Custom Formatter incorporating color coding for console logs."""

    PURPLE = "\x1b[35;1m"
    GREEN = "\x1b[32;1m"
    GREY = "\x1b[38;21m"
    YELLOW = "\x1b[33;21m"
    RED = "\x1b[31;21m"
    BOLD_RED = "\x1b[31;1m"
    BLUE = "\x1b[34;21m"
    RESET = "\x1b[0m"
    FORMAT = " >> %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREY + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class CustomLogger:
    _console_logger = None
    _file_logger = None

    @staticmethod
    def console_formatter():
        """Formatter for console logger"""
        return CustomFormatter()

    @staticmethod
    def file_formatter():
        """Formatter for file logger"""
        log_format = "%(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter

    @staticmethod
    def get_console_logger():
        if CustomLogger._console_logger is not None:
            return CustomLogger._console_logger

        console_logger = logging.getLogger("console_logger")
        console_logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(CustomLogger.console_formatter())
        console_logger.addHandler(console_handler)

        CustomLogger._console_logger = console_logger
        return console_logger

    @staticmethod
    def get_file_logger():
        if CustomLogger._file_logger is not None:
            return CustomLogger._file_logger

        file_logger = logging.getLogger("file_logger")
        file_logger.setLevel(logging.DEBUG)

        today = dt.datetime.today()
        filename = f"{today.month:02d}-{today.day:02d}-{today.year}.log"
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(CustomLogger.file_formatter())
        file_logger.addHandler(file_handler)
        CustomLogger._file_logger = file_logger
        return file_logger


file_logger = CustomLogger.get_file_logger()
console_logger = CustomLogger.get_console_logger()


def log(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    msg: str = "",
    console_logger_on: bool = True,
    file_logger_on: bool = True,
):
    """
    Logs a message with the specified logging level.

    Args:
        level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]): The logging level.
        msg (str, optional): The message to be logged. Defaults to "".
        console_logger_on (bool, optional): Flag indicating whether to log to the console. Defaults to True.
        file_logger_on (bool, optional): Flag indicating whether to log to a file. Defaults to True.
    """
    log_type_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_type = log_type_map.get(level, logging.INFO)

    if console_logger_on:
        console_logger.log(log_type, msg)
    if file_logger_on:
        file_logger.log(log_type, msg)
