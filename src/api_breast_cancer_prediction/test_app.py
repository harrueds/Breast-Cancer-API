"""
test.py

This script aims to validate the correct functioning of the REST API implemented in app.py.
Through HTTP requests made with the requests library, both the accessibility of the
server and the ability to generate predictions and handle erroneous inputs appropriately are verified.
"""

import requests
import json
import logging


# ====================================================================
# LOGGING SYSTEM CONFIGURATION
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/test_app_breast_cancer.log"),
        logging.StreamHandler(),
    ],
)


def test():

    logging.info("Starting REST API tests")

    # ====================================================================
    # 1. Test root endpoint
    # ====================================================================
    logging.info("=" * 60)
    res = requests.get("http://127.0.0.1:5000/")
    logging.info(f"GET /: {res.json()}")
    logging.info("=" * 60)

    # ====================================================================
    # 2. Test predict endpoint with valid data
    # ====================================================================
    example = {
        "features": [
            14.2,
            20.3,
            92.4,
            600.5,
            0.1,
            0.2,
            0.3,
            0.1,
            0.2,
            0.05,
            0.3,
            1.0,
            2.0,
            30.0,
            0.01,
            0.1,
            0.05,
            0.01,
            0.05,
            0.01,
            15.0,
            25.0,
            100.0,
            700.0,
            0.12,
            0.4,
            0.6,
            0.2,
            0.3,
            0.08,
        ]
    }

    res = requests.post("http://127.0.0.1:5000/predict", json=example)
    logging.info(f"POST /predict: {res.json()}")
    logging.info("=" * 60)

    # ====================================================================
    # 3. Test with input errors
    # ====================================================================

    # a) Incorrect key
    res = requests.post("http://127.0.0.1:5000/predict", json={"key_test": [1, 2, 3]})
    logging.info(f"POST /predict with error: {res.json()}")
    logging.info("=" * 60)

    # b) Incorrect data type
    res = requests.post(
        "http://127.0.0.1:5000/predict", json={"features": "incorrect_data"}
    )
    logging.info(f"POST /predict with error: {res.json()}")
    logging.info("=" * 60)

    # c) Empty JSON
    res = requests.post("http://127.0.0.1:5000/predict", json={})
    logging.info(f"POST /predict with error: {res.json()}")
    logging.info("=" * 60)

    # d) Corrupted data (mixed types)
    corrupted = {
        "features": [
            14.2,
            20.3,
            92.4,
            600.5,
            0.1,
            0.2,
            0.3,
            0.1,
            0.2,
            0.05,
            0.3,
            1.0,
            2.0,
            30.0,
            0.01,
            0.1,
            0.05,
            0.01,
            0.05,
            0.01,
            15.0,
            25.0,
            100.0,
            "a",
            0.12,
            0.4,
            0.6,
            0.2,
            0.3,
            0.08,
        ]
    }

    res = requests.post("http://127.0.0.1:5000/predict", json=corrupted)
    logging.info(f"POST /predict with error: {res.json()}")
    logging.info("=" * 60)

    # ====================================================================
    # 4. Tests completed successfully
    # ====================================================================
    logging.info("Tests completed successfully")
    logging.info("End of tests")
    logging.info("=" * 60)


if __name__ == "__main__":
    test()
