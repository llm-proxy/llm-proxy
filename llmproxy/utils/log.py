import datetime as dt
import logging
import sys
import time


class CustomLogger:
    _logger = None

    class CustomFormatter(logging.Formatter):
        """Custom Formatter incorporating color coding for console logs."""

        purple = "\x1b[35;1m"
        green = "\x1b[32;1m"
        grey = "\x1b[38;21m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        blue = "\x1b[34;21m"
        reset = "\x1b[0m"
        format = " >> %(message)s"

        FORMATS = {
            logging.DEBUG: grey + format + reset,
            logging.INFO: grey + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset,
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            return formatter.format(record)

    @staticmethod
    def file_formatter():
        """Formatter for file logger"""
        log_format = "%(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"
        return logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    @staticmethod
    def console_formatter():
        """Formatter for console logger with color coding"""
        return CustomLogger.CustomFormatter()

    @staticmethod
    def get_logger():
        if CustomLogger._logger is not None:
            return CustomLogger._logger

        # Create a parent logger
        CustomLogger._logger = logging.getLogger("parent_logger")
        CustomLogger._logger.setLevel(logging.DEBUG)

        # Child logger for file output
        file_logger = logging.getLogger("parent_logger.file_logger")
        today = dt.datetime.today()
        filename = f"{today.month:02d}-{today.day:02d}-{today.year}.log"

        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(CustomLogger.file_formatter())
        file_logger.addHandler(file_handler)
        file_logger.propagate = False

        # Child logger for terminal output
        console_logger = logging.getLogger("parent_logger.console_logger")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(CustomLogger.console_formatter())
        console_logger.addHandler(console_handler)
        console_logger.propagate = False

        return [file_logger, console_logger]


# Toggle
file_logger, console_logger = CustomLogger.get_logger()
