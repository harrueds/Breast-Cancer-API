import src.api_breast_cancer_prediction.training
import requests
import json
import logging


# ====================================================================
# LOGGING SYSTEM CONFIGURATION
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/main_breast_cancer.log"),
        logging.StreamHandler(),
    ],
)


# def run_training():
logging.info("Starting training...")
src.api_breast_cancer_prediction.training.train_model()
logging.info("Training complete.")


import src.api_breast_cancer_prediction.app
import src.api_breast_cancer_prediction.test_app


def run_app_and_check():
    logging.info("Starting app...")
    result = src.api_breast_cancer_prediction.app.run()
    logging.info("App complete.")

    # Simple verification app.py "is receiving"
    if result is None:
        logging.info("WARNING: app.py did not return any status.")
        return False

    logging.info("App returned:", result)
    return True


def run_test_app():
    logging.info("Starting test_app...")
    src.api_breast_cancer_prediction.test_app.test()
    logging.info("Test app complete.")


if __name__ == "__main__":
    logging.info("Type what you want to execute:")
    logging.info(" - 'execute'  -> run flask app")
    logging.info(" - 'exit'     -> exit the program")

    command = input("Your choice: ").strip().lower()

    while command not in ["execute", "exit"]:
        logging.info("Invalid command. Please type 'execute' or 'exit'.")
        command = input("Your choice: ").strip().lower()

    if command == "execute":
        logging.info("Executing flask app...")
        run_app_and_check()

    elif command == "exit":
        logging.info("Exiting the program.")

    logging.info("Henzo Alejandro Arrué Muñoz")
    logging.info("Version: 1.1.0")
    logging.info("Date: 2026-02")
