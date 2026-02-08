import logging
import os


def setup_logging(log_file: str):
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%d/%m/%Y - %H:%M:%S",
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{log_file}"),
            logging.StreamHandler(),
        ],
    )
