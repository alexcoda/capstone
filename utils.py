"""Utility functions."""
import logging


def get_logger(name=__name__, verbosity=2):
    """Generate a logger for use."""
    log_format_string = ' %(asctime)s %(name)-12s %(levelname)-8s\n%(message)s'

    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Handle the verbosity level
    levels = {0: logging.CRITICAL,
              1: logging.ERROR,
              2: logging.WARNING,
              3: logging.INFO,
              4: logging.DEBUG}
    level = levels[verbosity]
    logger.setLevel(level)
    return logger
