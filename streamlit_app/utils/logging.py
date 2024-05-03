import logging


def get_logger() -> logging.Logger:
    logger = logging.getLogger("redbox-streamlit")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s | %(name)s | %(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M",
    )

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
