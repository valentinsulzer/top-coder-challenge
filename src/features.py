import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the reimbursement model.
    Assumes df has 'trip_duration_days', 'miles_traveled', 'total_receipts_amount'.
    If 'expected_output' is present, it also engineers the target variable.
    """
    df_eng = df.copy()

    # Handle trips with 0 duration to avoid division by zero
    duration_safe = df_eng["trip_duration_days"].replace(0, 1)

    # Basic features
    df_eng["miles_per_day"] = df_eng["miles_traveled"] / duration_safe
    df_eng["receipts_per_day"] = df_eng["total_receipts_amount"] / duration_safe

    # Target variable engineering, if possible
    if "expected_output" in df_eng.columns:
        df_eng["reimbursement_per_day"] = df_eng["expected_output"] / duration_safe
        df_eng["log1p_reimbursement_per_day"] = np.log1p(
            df_eng["reimbursement_per_day"]
        )

    # Log1p features
    df_eng["log1p_miles_traveled"] = np.log1p(df_eng["miles_traveled"])
    df_eng["log1p_trip_duration_days"] = np.log1p(df_eng["trip_duration_days"])
    df_eng["log1p_total_receipts_amount"] = np.log1p(df_eng["total_receipts_amount"])
    df_eng["log1p_miles_per_day"] = np.log1p(df_eng["miles_per_day"])
    df_eng["log1p_receipts_per_day"] = np.log1p(df_eng["receipts_per_day"])

    # Polynomial features
    df_eng["miles_sq"] = df_eng["miles_traveled"] ** 2
    df_eng["receipts_sq"] = df_eng["total_receipts_amount"] ** 2
    df_eng["duration_sq"] = df_eng["trip_duration_days"] ** 2

    # Interaction features
    df_eng["miles_x_duration"] = df_eng["miles_traveled"] * df_eng["trip_duration_days"]
    df_eng["receipts_x_duration"] = (
        df_eng["total_receipts_amount"] * df_eng["trip_duration_days"]
    )
    df_eng["miles_x_receipts"] = (
        df_eng["miles_traveled"] * df_eng["total_receipts_amount"]
    )

    main_features = [
        "log1p_miles_per_day",
        "log1p_receipts_per_day",
        "miles_per_day",
        "receipts_per_day",
        "trip_duration_days",
    ]

    # Pairwise products
    for i in range(len(main_features)):
        for j in range(i + 1, len(main_features)):
            f1 = main_features[i]
            f2 = main_features[j]
            col_name = f"{f1}_x_{f2}"
            df_eng[col_name] = df_eng[f1] * df_eng[f2]

    return df_eng
