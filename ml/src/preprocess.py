"""
InfraGuardAI Data Preprocessing

Loads infrastructure_data.csv, prepares numerical and categorical
features, performs a stratified train/test split, and returns the
feature schema used by the ML model.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


CURRENT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    CURRENT_DIR.parent
    / "data"
    / "infrastructure_data.csv"
)


NUMERICAL_COLS = [
    "age_years",
    "condition_score",
    "complaint_count",
    "failure_count",
    "days_since_maintenance",
    "inspection_score",
]


CATEGORICAL_COLS = [
    "asset_type",
    "usage_level",
    "environmental_risk",
]


TARGET_COL = "maintenance_required"


def load_dataset():
    """
    Load the infrastructure dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "The infrastructure dataset is empty."
        )

    return df


def validate_dataset(df):
    """
    Validate that all required columns exist.
    """

    required_columns = (
        NUMERICAL_COLS
        + CATEGORICAL_COLS
        + [TARGET_COL]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns:\n"
            + ", ".join(missing_columns)
        )


def clean_dataset(df):
    """
    Clean and normalize the raw dataset.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Numerical columns
    # ---------------------------------------------------------

    for column in NUMERICAL_COLS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Categorical columns
    # ---------------------------------------------------------

    for column in CATEGORICAL_COLS:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    if df[TARGET_COL].dtype == object:

        target_mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }

        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_mapping)
        )

    else:

        df[TARGET_COL] = pd.to_numeric(
            df[TARGET_COL],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Remove invalid rows
    # ---------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=(
            NUMERICAL_COLS
            + CATEGORICAL_COLS
            + [TARGET_COL]
        )
    )

    after = len(df)

    if before != after:
        print(
            f"Removed {before - after} invalid rows."
        )

    # ---------------------------------------------------------
    # Force target to integer
    # ---------------------------------------------------------

    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(int)
    )

    return df


def prepare_features(df):
    """
    Convert raw columns into model-ready features.
    """

    X = df[
        NUMERICAL_COLS
        + CATEGORICAL_COLS
    ].copy()

    y = df[TARGET_COL].copy()

    # One-hot encode categorical columns.
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLS,
        drop_first=False,
    )

    # Make every feature numeric.
    X = X.astype(float)

    feature_names = list(
        X.columns
    )

    return X, y, feature_names


def load_and_preprocess(
    test_size=0.20,
    random_state=42,
):
    """
    Complete preprocessing pipeline.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    feature_names
    """

    print(
        f"Loading dataset from:\n{DATA_PATH}"
    )

    df = load_dataset()

    print(
        f"Raw dataset shape: {df.shape}"
    )

    validate_dataset(df)

    df = clean_dataset(df)

    print(
        f"Clean dataset shape: {df.shape}"
    )

    X, y, feature_names = (
        prepare_features(df)
    )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        "Target distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # Validate target
    # ---------------------------------------------------------

    unique_targets = sorted(
        y.unique().tolist()
    )

    if len(unique_targets) != 2:
        raise ValueError(
            "The target column must contain exactly "
            f"two classes. Found: {unique_targets}"
        )

    # ---------------------------------------------------------
    # Stratified train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    print()
    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    print(
        f"Total features: {len(feature_names)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )


if __name__ == "__main__":

    print("=" * 60)
    print("       INFRAGUARDAI PREPROCESSING TEST")
    print("=" * 60)

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names,
        ) = load_and_preprocess()

        print()
        print("PREPROCESSING SUCCESSFUL")
        print(
            "X_train:",
            X_train.shape
        )

        print(
            "X_test:",
            X_test.shape
        )

        print(
            "y_train:",
            y_train.shape
        )

        print(
            "y_test:",
            y_test.shape
        )

        print(
            "Features:",
            len(feature_names)
        )

    except Exception as error:

        print()
        print("PREPROCESSING FAILED")
        print(
            type(error).__name__
        )
        print(
            str(error)
        )