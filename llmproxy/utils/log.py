import datetime as dt
import logging
import time
import sys


class CustomLogger:
    _logger = None
    _show_details = True

    class CustomFormatter(logging.Formatter):
        """Logging Levels
        1. Debug: Used when diagnosing problems
        2. Info: Confirm that things are working as expected
        3. Warning: Indication that something unexpected happened, or something that may lead to a bigger issue
        4. Error: A serious problem; the software wasn't able to do something
        5. Critical: A serious error; program may be unable to continue running
        """

        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        green = "\x1b[32;20m"

        log_format = " %(name)s: %(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"

        FORMATS = {
            logging.DEBUG: grey + log_format + reset,
            logging.INFO: grey + log_format + reset,
            logging.WARNING: yellow + log_format + reset,
            logging.ERROR: red + log_format + reset,
            logging.CRITICAL: bold_red + log_format + reset,
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            if not CustomLogger._show_details:
                #log_fmt = " %(name)s: %(asctime)s [%(levelname)s] >> %(message)s"
                log_fmt = ">> %(message)s"
            formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            return formatter.format(record)

    @classmethod
    def get_logger(cls, logger_name: str = "LOGGER", show_details: bool = True):
        # If logger exist return the logger else create new logger and return
        if cls._logger is not None:
            return cls._logger

        today = dt.datetime.today()
        filename = f"{today.month:02d}-{today.day:02d}-{today.year}.log"

        log_format = " %(name)s: %(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"

        # Setup logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(level=logging.DEBUG)

        cls._show_details = show_details

        cls._configure_handlers(logger, filename, log_format)

        cls._logger = logger
        return logger

    @classmethod
    def _configure_handlers(cls, logger, filename, log_format):
        # Remove existing handlers (if any)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(cls.CustomFormatter())
        logger.addHandler(ch)

        # File handler
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    @classmethod
    def loading_animation(cls):
        print("LOADING...")
        animation = ["[■■□□□□□□□□□□□□□□□□□□]", "[■■■■□□□□□□□□□□□□□□□□]",
                     "[■■■■■■□□□□□□□□□□□□□□]", "[■■■■■■■■□□□□□□□□□□□□]",
                     "[■■■■■■■■■■□□□□□□□□□□]", "[■■■■■■■■■■■■□□□□□□□□]",
                     "[■■■■■■■■■■■■■■□□□□□□]", "[■■■■■■■■■■■■■■■■□□□□]",
                     "[■■■■■■■■■■■■■■■■■■□□]", "[■■■■■■■■■■■■■■■■■■■■]"]

        for i in range(len(animation)):
            time.sleep(0.05)
            sys.stdout.write(
                "\r" + cls.CustomFormatter.green + animation[i % len(animation)] + cls.CustomFormatter.reset)
            sys.stdout.flush()
        print("\n")


# Toggle
logger = CustomLogger.get_logger("LOG", show_details = False)