import logging
import sys

# Package-legel logger
logger = logging.getLogget("thermohl")

# Default configuration
def configure_logger(level=logging.INFO, format_string=None): 
    """
    Configure the thermohl logger.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_string = Custom format_string for log messages
    """
    if format_string is None:
        format_string = "%f(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
    logger.setLevel(level)

# Setup default configurations
if not logger.handler:
    configure_logger()
