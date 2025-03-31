import logging

def setup_logger():
    """Sets up the logging configuration."""
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(logging.DEBUG)

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # You can change this to DEBUG for more details

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler("fraud_detection.log")
    file_handler.setLevel(logging.DEBUG)

    # Define log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
