import os

import joblib
import pandas as pd

from .risk_engine import calculate_risk
from .priority_engine import calculate_priority


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "maintenance_model.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "feature_columns.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    return joblib.load(MODEL_PATH)


def load_feature_columns():
    return joblib.load(FEATURES_PATH)


model = load_model()
feature_columns = load_feature_columns()


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_FIELDS = [
    "asset_type",
    "age_years",
    "condition_score",
    "complaint_count",
    "failure_count",
    "days_since_maintenance",
    "inspection_score",
    "environmental_risk",
    "usage_level",
]


def validate_input(data):

    if not isinstance(data, dict):
        raise ValueError(
            "Prediction input must be a dictionary."
        )

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing_fields:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing_fields)
        )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(data):

    validate_input(data)

    dataframe = pd.DataFrame(
        [data]
    )

    categorical_columns = [
        "asset_type",
        "environmental_risk",
        "usage_level",
    ]

    dataframe = pd.get_dummies(
        dataframe,
        columns=categorical_columns,
        drop_first=False
    )

    dataframe = dataframe.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return dataframe


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanation(data):

    factors = []

    condition_score = float(
        data.get("condition_score", 50)
    )

    age_years = float(
        data.get("age_years", 0)
    )

    complaint_count = int(
        data.get("complaint_count", 0)
    )

    failure_count = int(
        data.get("failure_count", 0)
    )

    days_since_maintenance = int(
        data.get(
            "days_since_maintenance",
            0
        )
    )

    inspection_score = float(
        data.get("inspection_score", 50)
    )

    environmental_risk = data.get(
        "environmental_risk",
        "Low"
    )

    usage_level = data.get(
        "usage_level",
        "Medium"
    )


    if condition_score <= 40:

        factors.append(
            "Poor infrastructure condition"
        )

    elif condition_score <= 60:

        factors.append(
            "Moderate infrastructure condition"
        )


    if age_years >= 15:

        factors.append(
            "High asset age"
        )

    elif age_years >= 10:

        factors.append(
            "Elevated asset age"
        )


    if complaint_count >= 10:

        factors.append(
            "High number of citizen complaints"
        )

    elif complaint_count >= 5:

        factors.append(
            "Increasing complaint activity"
        )


    if failure_count >= 3:

        factors.append(
            "Repeated infrastructure failures"
        )

    elif failure_count >= 1:

        factors.append(
            "Previous infrastructure failures"
        )


    if days_since_maintenance >= 300:

        factors.append(
            "Maintenance overdue for a long period"
        )

    elif days_since_maintenance >= 180:

        factors.append(
            "Maintenance interval is elevated"
        )


    if inspection_score <= 40:

        factors.append(
            "Poor inspection score"
        )

    elif inspection_score <= 60:

        factors.append(
            "Moderate inspection score"
        )


    if environmental_risk == "High":

        factors.append(
            "High environmental risk"
        )

    elif environmental_risk == "Medium":

        factors.append(
            "Moderate environmental risk"
        )


    if usage_level == "High":

        factors.append(
            "High infrastructure usage"
        )


    if not factors:

        factors.append(
            "No major risk indicators detected"
        )


    return factors


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_asset(data):

    validate_input(data)

    input_dataframe = prepare_input(data)


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_dataframe
    )[0]


    # --------------------------------------------------------
    # MAINTENANCE PROBABILITY
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_dataframe
        )[0]

        classes = list(
            model.classes_
        )

        if 1 in classes:

            maintenance_index = classes.index(1)

            maintenance_probability = float(
                probabilities[
                    maintenance_index
                ]
            )

        else:

            maintenance_probability = (
                1.0
                if int(prediction) == 1
                else 0.0
            )

    else:

        maintenance_probability = (
            1.0
            if int(prediction) == 1
            else 0.0
        )


    maintenance_probability = round(
        maintenance_probability * 100,
        2
    )


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk = calculate_risk(
        maintenance_probability
    )


    if isinstance(risk, dict):

        risk_score = risk.get(
            "score",
            risk.get(
                "risk_score",
                0
            )
        )

        risk_category = risk.get(
            "category",
            risk.get(
                "risk_category",
                "Low"
            )
        )

    else:

        risk_score = float(risk)

        if risk_score <= 30:

            risk_category = "Low"

        elif risk_score <= 60:

            risk_category = "Medium"

        elif risk_score <= 80:

            risk_category = "High"

        else:

            risk_category = "Critical"


    risk_score = round(
        float(risk_score),
        2
    )


    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    priority = calculate_priority(
        risk_score,
        data.get("asset_type"),
        int(
            data.get(
                "complaint_count",
                0
            )
        ),
        int(
            data.get(
                "failure_count",
                0
            )
        ),
        data.get(
            "usage_level",
            "Medium"
        )
    )


    if not isinstance(priority, dict):

        priority = {
            "tier": "P4",
            "score": 0,
            "sla_days": 30,
            "action": "Routine monitoring"
        }


    priority_tier = priority.get(
        "tier",
        priority.get(
            "priority",
            "P4"
        )
    )

    priority_score = priority.get(
        "score",
        priority.get(
            "priority_score",
            0
        )
    )

    sla_days = priority.get(
        "sla_days",
        30
    )

    action = priority.get(
        "action",
        "Routine monitoring"
    )


    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    explanation = generate_explanation(
        data
    )


    # --------------------------------------------------------
    # RECOMMENDED ACTION
    # --------------------------------------------------------

    if risk_category == "Critical":

        recommended_action = (
            "Immediate inspection and urgent "
            "maintenance required."
        )

    elif risk_category == "High":

        recommended_action = (
            "Schedule priority inspection "
            "and maintenance."
        )

    elif risk_category == "Medium":

        recommended_action = (
            "Plan preventive maintenance "
            "and continue monitoring."
        )

    else:

        recommended_action = (
            "Continue routine monitoring."
        )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "maintenance_required": bool(
            int(prediction) == 1
        ),

        "maintenance_probability": (
            maintenance_probability
        ),

        "risk": {

            "score": risk_score,

            "category": risk_category

        },

        "priority": {

            "tier": priority_tier,

            "score": priority_score,

            "sla_days": sla_days,

            "action": action

        },

        "explanation": explanation,

        "recommended_action": (
            recommended_action
        )

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_asset = {

        "asset_type": "Road",

        "age_years": 12,

        "condition_score": 48,

        "complaint_count": 17,

        "failure_count": 4,

        "days_since_maintenance": 290,

        "inspection_score": 52,

        "environmental_risk": "High",

        "usage_level": "High"

    }


    result = predict_asset(
        test_asset
    )


    print()
    print("=" * 60)
    print("INFRA GUARD AI - PREDICTION TEST")
    print("=" * 60)

    print(
        "Maintenance Required:",
        result["maintenance_required"]
    )

    print(
        "Maintenance Probability:",
        f'{result["maintenance_probability"]}%'
    )

    print(
        "Risk Score:",
        result["risk"]["score"]
    )

    print(
        "Risk Category:",
        result["risk"]["category"]
    )

    print(
        "Priority:",
        result["priority"]["tier"]
    )

    print(
        "Priority Score:",
        result["priority"]["score"]
    )

    print(
        "SLA:",
        result["priority"]["sla_days"],
        "days"
    )

    print(
        "Action:",
        result["priority"]["action"]
    )

    print()
    print("Risk Factors:")

    for factor in result["explanation"]:

        print(
            " -",
            factor
        )

    print("=" * 60)