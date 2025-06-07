import sys
import os
import pandas as pd
import numpy as np
from src.lightgbm_model import LightGBMReimbursementModel
from src.features import engineer_features

FEATURES = [
    "trip_duration_days",
    "miles_traveled",
    "total_receipts_amount",
    "miles_per_day",
    "receipts_per_day",
    "log1p_miles_per_day",
    "log1p_receipts_per_day",
    "log1p_miles_traveled",
    "log1p_trip_duration_days",
    "log1p_total_receipts_amount",
    "miles_sq",
    "receipts_sq",
    "duration_sq",
    "miles_x_duration",
    "receipts_x_duration",
    "miles_x_receipts",
    "log1p_miles_per_day_x_log1p_receipts_per_day",
    "log1p_miles_per_day_x_miles_per_day",
    "log1p_miles_per_day_x_receipts_per_day",
    "log1p_miles_per_day_x_trip_duration_days",
    "log1p_receipts_per_day_x_miles_per_day",
    "log1p_receipts_per_day_x_receipts_per_day",
    "log1p_receipts_per_day_x_trip_duration_days",
    "miles_per_day_x_receipts_per_day",
    "miles_per_day_x_trip_duration_days",
    "receipts_per_day_x_trip_duration_days",
]


def _load_model():
    # Load from CSV with features
    csv_path = os.path.join(
        os.path.dirname(__file__), "../public_cases_with_features.csv"
    )
    df = pd.read_csv(csv_path)

    # Replace inf/-inf with NaN and drop rows where target is NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log1p_reimbursement_per_day"], inplace=True)

    X = df[FEATURES]
    y = df["log1p_reimbursement_per_day"]
    model = LightGBMReimbursementModel()
    model.fit(X, y)
    return model, X.columns.tolist()


def calculate_reimbursement(model, feature_cols, **kwargs) -> float:
    """Calculate reimbursement amount based on input features."""
    # Engineer features from raw inputs
    X = engineer_features(pd.DataFrame([kwargs]))

    # Ensure all columns match training
    for col in feature_cols:
        if col not in X:
            X[col] = 0
    X = X[feature_cols]

    # Predict log of reimbursement per day
    log_pred = model.predict(X)[0]

    # Convert back to total reimbursement
    reimbursement_per_day = np.expm1(log_pred)
    duration = X["trip_duration_days"].iloc[0]
    total_reimbursement = reimbursement_per_day * duration

    return round(float(total_reimbursement), 2)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python src/main.py <trip_duration_days> <miles_traveled> <total_receipts_amount>"
        )
        sys.exit(1)

    try:
        trip_duration_days = int(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        print(
            "Error: Invalid input types. Please provide integers for days and miles, and a float for receipts."
        )
        sys.exit(1)

    try:
        model, feature_cols = _load_model()
        reimbursement_amount = calculate_reimbursement(
            model,
            feature_cols,
            trip_duration_days=trip_duration_days,
            miles_traveled=miles_traveled,
            total_receipts_amount=total_receipts_amount,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(reimbursement_amount)


if __name__ == "__main__":
    main()
