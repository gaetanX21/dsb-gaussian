# This script sets up a logging system with both console and file handlers.
# It defines a `ColorFormatter` class to add color to console log messages based on their severity level.
# The `setup_logger` function configures a logger with the specified name, verbosity level, and log file.
# The logger outputs colored messages to the console and plain messages to a log file.

import logging
from os.path import join


class ColorFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        "DEBUG": "\033[92m",  # Green
        "INFO": "\033[94m",   # Blue
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"  # Reset to default

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{log_color}{record.msg}{self.RESET}"  # Wrap message in color
        return super().format(record)
    

def setup_logger(logger_name: str, verbose_level: int, log_file: str):
    """Configures and returns a logger with both console and file handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(verbose_level)

    # Console handler with color
    console = logging.StreamHandler()
    console.setLevel(verbose_level) # console verbosity level is set by the user
    color_formatter = ColorFormatter('%(levelname)s - %(message)s')
    console.setFormatter(color_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO) # only info in the log
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger