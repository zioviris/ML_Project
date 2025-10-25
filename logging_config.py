import logging

def setup_logger():
    """Sets up the logging configuration."""
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(logging.DEBUG)


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  #  DEBUG for more details


    file_handler = logging.FileHandler("fraud_detection.log")
    file_handler.setLevel(logging.DEBUG)

   
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
