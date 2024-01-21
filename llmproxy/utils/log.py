import datetime as dt
import logging


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
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(logger_name: str = "LOGGER"):
    today = dt.datetime.today()
    # :02d format gives us two digits always: ex. 09, 08, 12,...
    filename = f"{today.month:02d}-{today.day:02d}-{today.year}.log"

    log_format = " %(name)s: %(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"

    # Setup logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)

    # if setting logging level does not work by default you can do the following
    # for handler in logging.root.handlers:
    #     logging.root.removeHandler(handler)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    #  Setup file handler
    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.WARNING)

    # Formatter for logs
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = get_logger("LOG")
