# Breast Cancer Prediction REST API

## Machine Learning · MLOps · Python · Docker · CI/CD

[![wakatime](https://wakatime.com/badge/user/8bfb91a4-b57e-4540-bf32-05c8afcd58a3/project/2c0756c4-1677-488d-8e99-e8fd044b0950.svg)](https://wakatime.com/badge/user/8bfb91a4-b57e-4540-bf32-05c8afcd58a3/project/2c0756c4-1677-488d-8e99-e8fd044b0950)

## **Project Status:** Active Development (Pre-Production)

- This project is under active development and intended as a production-oriented ML engineering showcase.
- The API contract is considered stable, but internal components may evolve.

This repository showcases a production-oriented Machine Learning REST API for breast cancer prediction, built with **Python** and **Flask**, following modern ML engineering and MLOps best practices.

The project is intentionally designed to reflect real-world ML workflows, emphasizing reproducibility, clean architecture, automated testing, and deployment readiness rather than notebook-driven experimentation.

---

## Project Highlights

- End-to-end ML pipeline: **training → persistence → inference via REST API**
- Production-ready structure using the **`src/` layout** (PEP-compliant, import-safe)
- Reproducible environments with **`uv`** and locked dependencies (`pyproject.toml` + `uv.lock`)
- Centralized logging configuration via `logging_config.py`
- API startup protection: prevents serving predictions without a trained model
- Automated testing with integration coverage
- Containerized deployment with Docker
- CI/CD automation with GitHub Actions

This project is suitable as a portfolio reference for **ML Engineer / Applied Data Scientist** roles.

---

## Technical Stack

- **Language:** Python 3.9+
- **API Framework:** Flask
- **ML Stack:** scikit-learn
- **Dependency Management:** uv (`pyproject.toml` + `uv.lock`)
- **Testing:** Unit & integration tests
- **Containerization:** Docker
- **CI/CD:** GitHub Actions (not yet implemented)
- **Architecture:** `src/`-based Python package layout

---

## Architecture Overview

```bash
Training Pipeline
└── training.py
    ├── Data loading & preprocessing
    ├── Model training
    ├── Model evaluation
    └── Model persistence (models/model_breast_cancer.pkl)

Inference Pipeline
└── app.py
    ├── Model loading (on first prediction request)
    ├── Strict numerical input validation
    ├── Prediction endpoint (/predict)
    └── JSON response

Observability & Delivery
├── logging_config.py → Centralized logging setup
├── test_app.py       → Automated tests
├── Dockerfile        → Reproducible builds
└── CI/CD pipeline    → Continuous validation
```

The design mirrors how ML systems are typically deployed in professional MLOps environments.

## Project Structure

```bash
Breast-Cancer-API/
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── README.md
├── src/
│   └── api_breast_cancer_prediction/
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── training.py
│       ├── test_app.py
│       └── logging_config.py
│
├── models/
│   └── model_breast_cancer.pkl
│
├── logs/
│   ├── training_breast_cancer.log
│   ├── app_breast_cancer.log
│   └── main_breast_cancer.log
│
├── examples/
│   ├── invalid_type.json
│   ├── missing_features.json
│   ├── missing_key.json
│   ├── valid_request.json
│   └── wrong_features_type.json
│
└── .github/workflows/
    └── ci-cd.yml
```

## Environment & Reproducibility

Dependencies and virtual environments are managed using uv, ensuring:

- Deterministic builds
- Fast dependency resolution
- Clean separation from system Python

### Install dependencies

```bash
uv sync
```

All commands should be executed inside the managed environment:

```bash
uv run python ...
```

## Model Training

Train the supervised ML model and persist it into models/:

```bash
uv run python -m src.api_breast_cancer_prediction.training
```

Training produces:

- A trained classification model

- A saved artifact:

```bash
models/model_breast_cancer.pkl
```

- Training logs:

```bash
logs/training_breast_cancer.log
```

## Running the API

Start the Flask inference service:

```bash
uv run python -m src.api_breast_cancer_prediction.app
```

Default endpoint:

```bash
http://127.0.0.1:5000
```

Health check:

```bash
curl http://127.0.0.1:5000
```

## Prediction Endpoint

The main endpoint is:

```text
POST /predict
```

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [14.2, 20.1, 92.3, 600.1, 0.11]}'
```

## Example Payloads for API Validation

This repository includes a complete set of JSON request payloads designed to validate the API input contract.

These files serve as:

- Practical usage examples for API consumers

- Contract testing references

- Input validation edge-case coverage

## All payloads are located in

```bash
examples/
```

### Available Test Payloads

|File|Purpose|Expected Result|
|-|-|-|
|valid_request.json|Correct input format|Successful prediction|
|missing_features.json|Missing feature values inside array|400 Validation Error|
|missing_key.json|Entire features key omitted|400 Bad Request|
|wrong_features_type.json|features provided as wrong container type|400 Bad Request|
|invalid_type.json|Non-numeric value inside feature list|400 Validation Error|

### Running Manual Requests with curl

You can test the API manually using:

- **Valid Request**

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @examples/valid_request.json
```

- **Missing Feature Values**

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @examples/missing_features.json
```

- **Missing Required Key (features)**

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @examples/missing_key.json
```

- **Wrong features Container Type**

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @examples/wrong_features_type.json
```

- **Invalid Feature Value Type (Non-Numeric)**

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @examples/invalid_type.json
```

### Why These Payloads Matter

Strict input validation is critical in production ML systems.

These payloads ensure:

- The API rejects malformed requests early
- The model never receives invalid tensors
- Consumers understand the required schema
- Edge cases are reproducible and testable
- This mirrors real-world ML deployment practices where prediction services must enforce a stable contract.

## Model Availability Behavior

The API requires a trained model artifact:

```bash
models/model_breast_cancer.pkl
```

If the model file is missing:

- The API will log an error

- The service will exit immediately:

```bash
You must run training.py before starting the API.
```

This ensures prediction endpoints are never served without a valid trained model.

## Logging & Observability

All components share a centralized logging system via:

```text
logging_config.py
```

Logs are written both to console and the logs/ directory:

- `logs/app_breast_cancer.log`
- `logs/training_breast_cancer.log`
- `logs/main_breast_cancer.log`

This mirrors production observability practices.

## Testing Strategy

Run automated tests locally:

```bash
uv run python -m src.api_breast_cancer_prediction.test_app
```

Tests validate:

- Core inference logic
- API endpoint behavior
- Input validation

Tests are also executed automatically in CI.

## Docker Deployment

Build the Docker image:

```bash
docker build -t breast_cancer_api .
```

Run the container:

```bash
docker run -d -p 5000:5000 breast_cancer_api
```

The container encapsulates:

- The trained model
- The API runtime
- Locked dependencies

## Version History

### v1.2.0 (2026-02)

- Centralized logging configuration (logging_config.py)
- Improved API robustness when model artifact is missing
- Cleaner startup behavior aligned with production services

### v1.1.0 (2026-01)

- Migrated to src/ layout
- Switched dependency management from pip to uv
- Improved reproducibility and packaging structure

## Author

### Henzo Alejandro Arrué Muñoz

- **Data Scientist & Machine Learning Practitioner**

### Email

- [harrue.ds@gmail.com](mailto:harrue.ds@gmail.com)
- [henzo.arruemu@itacademy.cl](mailto:henzo.arruemu@itacademy.cl)

### Profiles

- GitHub: [https://github.com/harrueds](https://github.com/harrueds)
- LinkedIn: [https://www.linkedin.com/in/henzo-arrué-muñoz/](https://www.linkedin.com/in/henzo-arrué-muñoz/)
- Dev.to: [https://dev.to/harrueds](https://dev.to/harrueds)
