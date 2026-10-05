"""
InfraGuardAI Model Training

Trains the Random Forest maintenance prediction model
and saves the model + feature schema into ml/models/.
"""

from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

try:
    from .preprocess import load_and_preprocess
except ImportError:
    from preprocess import load_and_preprocess


CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR.parent / "models"


def train_model():

    print("=" * 60)
    print("       INFRAGUARDAI MODEL TRAINING")
    print("=" * 60)

    print()
    print("Loading and preprocessing dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = load_and_preprocess()

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print(
        f"Features          : {len(feature_names)}"
    )

    print()

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training Random Forest...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print()
    print("=" * 60)
    print("              MODEL PERFORMANCE")
    print("=" * 60)

    print()
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    print(
        "ROC-AUC:",
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    confusion = confusion_matrix(
        y_test,
        predictions,
    )

    print()
    print("Confusion Matrix")

    print(
        "True Negatives :",
        confusion[0][0]
    )

    print(
        "False Positives:",
        confusion[0][1]
    )

    print(
        "False Negatives:",
        confusion[1][0]
    )

    print(
        "True Positives :",
        confusion[1][1]
    )

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODELS_DIR
        / "maintenance_model.pkl"
    )

    features_path = (
        MODELS_DIR
        / "feature_columns.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        feature_names,
        features_path,
    )

    print()
    print("=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)

    print()
    print(
        "Model:",
        model_path
    )

    print(
        "Features:",
        features_path
    )

    print()
    print("Training pipeline finished successfully.")


if __name__ == "__main__":
    train_model()