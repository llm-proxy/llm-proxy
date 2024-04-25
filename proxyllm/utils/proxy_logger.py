import datetime as dt
import logging
import sys
from logging import NullHandler
from typing import Literal


class CustomFormatter(logging.Formatter):
    """
    Custom logging Formatter class that adds color coding to log messages for console output,
    enhancing the visibility and differentiation of log levels.

    Attributes:
        Format strings for different logging levels with ANSI color codes.
    """

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
        """
        Formats the log messages based on log level.

        Args:
            record (logging.LogRecord): Log record to be formatted.

        Returns:
            str: Formatted log message with color codes based on the level of the log.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class CustomLogger:
    """
    Facilitates the creation and retrieval of customized console and file loggers.

    This class ensures that there is only one instance of console and file loggers
    throughout the application, following the singleton pattern.
    """

    _console_logger = None
    _file_logger = None

    @staticmethod
    def console_formatter():
        """
        Returns a custom logging formatter for console outputs with color coding.
        """
        return CustomFormatter()

    @staticmethod
    def file_formatter():
        """
        Returns a logging formatter for file outputs including timestamps, log level, and message details.
        """
        log_format = "%(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter

    @staticmethod
    def get_console_logger():
        """
        Retrieves the singleton instance of the console logger.

        Returns:
            logging.Logger: The singleton console logger instance.
        """
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
        """
        Retrieves the singleton instance of the file logger.

        Returns:
            logging.Logger: The singleton file logger instance with configured file handler and formatter.
        """
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


# Prevent external loggers from using our custom loggers
null_handler = NullHandler()
logging.getLogger().addHandler(null_handler)

file_logger = CustomLogger.get_file_logger()
console_logger = CustomLogger.get_console_logger()


def log(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    msg: str = "",
    console_logger_on: bool = True,
    file_logger_on: bool = True,
    color: Literal[
        "PURPLE",
        "GREEN",
        "GREY",
        "YELLOW",
        "RED",
        "BOLD_RED",
        "BLUE",
        "RESET",
        "FORMAT",
    ] = "GREY",
):
    """
    Logs a message with the specified logging level.

    Args:
        level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]): The logging level.
        msg (str, optional): The message to be logged. Defaults to "".
        console_logger_on (bool, optional): Flag indicating whether to log to the console. Defaults to True.
        file_logger_on (bool, optional): Flag indicating whether to log to a file. Defaults to True.
        color (Literal["PURPLE", "GREEN", "GREY", "YELLOW", "RED", "BOLD_RED", "BLUE", "RESET", "FORMAT"], optional):
        The color to be applied to the log message. Defaults to "GREY".
    """
    log_type_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_color_map = {
        "PURPLE": CustomFormatter.PURPLE,
        "GREEN": CustomFormatter.GREEN,
        "GREY": CustomFormatter.GREY,
        "YELLOW": CustomFormatter.YELLOW,
        "RED": CustomFormatter.RED,
        "BOLD_RED": CustomFormatter.BOLD_RED,
        "BLUE": CustomFormatter.BLUE,
        "RESET": CustomFormatter.RESET,
        "FORMAT": CustomFormatter.FORMAT,
    }

    log_type = log_type_map.get(level, logging.INFO)
    log_color = log_color_map.get(color, CustomFormatter.GREY)

    formatted_msg = msg
    if color:
        formatted_msg = f"{log_color}{msg}{log_color}"

    if console_logger_on:
        console_logger.log(log_type, formatted_msg)
    if file_logger_on:
        file_logger.log(log_type, msg)
