from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.api_breast_cancer_prediction.logging_config import setup_logging

import joblib
import logging
import os


def train_model():
    os.makedirs("models", exist_ok=True)  # create models directory if not present
    # ============================================================================
    # 1. DATASET LOADING
    # ============================================================================
    # Loads the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.
    #
    # Variables:
    #   X: Feature matrix (569 samples × 30 features)
    #   y: Target vector with class labels (0 = malignant, 1 = benign)
    #
    # The dataset is loaded directly in NumPy array format for processing.
    X, y = load_breast_cancer(return_X_y=True)
    logging.info("=" * 60)
    logging.info(
        "Breast Cancer Wisconsin Dataset loaded: %d samples, %d features",
        X.shape[0],
        X.shape[1],
    )
    logging.info("=" * 60)

    # ============================================================================
    # 2. SPLITTING DATA INTO TRAINING AND TEST SETS
    # ============================================================================
    # Splits the data into training (80%) and test (20%) sets.
    #
    # Parameters:
    #   test_size=0.2: Reserves 20% of the data for model evaluation
    #   random_state=42: Random seed to ensure reproducibility
    #
    # The split is random but reproducible, allowing consistent comparisons
    # across different script executions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info(
        "Data split - Training: %d samples, Test: %d samples",
        X_train.shape[0],
        X_test.shape[0],
    )

    # ============================================================================
    # 3. MODEL TRAINING
    # ============================================================================
    # Initializes and trains a Logistic Regression model.
    #
    # Hyperparameters:
    #   max_iter=5000: Maximum number of iterations for the optimizer (LBFGS by default)
    #                  Value increased from default (100) to ensure convergence
    #
    # The fit() method adjusts the model coefficients through iterative optimization,
    # minimizing the logistic loss function (log-loss) on the training data.
    model = LogisticRegression(max_iter=5000)
    logging.info("Starting Logistic Regression model training...")

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))

    logging.info("=" * 60)
    logging.info("Model trained successfully")
    logging.info("=" * 60)
    logging.info(f"Accuracy={accuracy:.4f}")
    logging.info(f"F1-Score={f1:.4f}")
    logging.info(f"Precision={precision:.4f}")
    logging.info(f"Recall={recall:.4f}")
    logging.info("=" * 60)

    # ============================================================================
    # 4. TRAINED MODEL SERIALIZATION
    # ============================================================================
    # Saves the trained model to disk using joblib for later use.
    #
    # Details:
    #   Format: Optimized pickle for NumPy/scikit-learn objects
    #   File: model.pkl
    #   Purpose: Allows model reuse in production without retraining
    #
    # The serialized model can be loaded later with joblib.load('model.pkl')
    # to make predictions on new data.
    joblib.dump(model, "models/model_breast_cancer.pkl")
    logging.info("Model serialized and saved as 'model_breast_cancer.pkl' in 'models/'")

    # ============================================================================
    # 5. COMPLETION CONFIRMATION
    # ============================================================================
    # Records successful completion of the training pipeline.
    logging.info("=" * 60)
    logging.info("Training pipeline completed successfully")
    logging.info("Model available at: models/model_breast_cancer.pkl")
    logging.info("Logs available at: logs/training_breast_cancer.log")
    logging.info("=" * 60)


if __name__ == "__main__":
    setup_logging("training_breast_cancer.log")
    train_model()
