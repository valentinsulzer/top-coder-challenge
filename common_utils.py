import pandas as pd
import numpy as np
import os
from human_model import output_model


FEATURES = [
    "trip_duration_days",
    "miles_traveled",
    "total_receipts_amount",
    "miles_per_day",
    "receipts_per_day",
    # "log1p_miles_per_day",
    # "log1p_receipts_per_day",
    # "log1p_miles_traveled",
    # "log1p_trip_duration_days",
    # "log1p_total_receipts_amount",
    # "miles_sq",
    # "receipts_sq",
    # "duration_sq",
    "miles_x_duration",
    "receipts_x_duration",
    "miles_x_receipts",
    "miles_per_receipt",
    "receipts_per_mile",
    # "log1p_miles_per_day_x_log1p_receipts_per_day",
    # "log1p_miles_per_day_x_miles_per_day",
    # "log1p_miles_per_day_x_receipts_per_day",
    # "log1p_miles_per_day_x_trip_duration_days",
    # "log1p_receipts_per_day_x_miles_per_day",
    # "log1p_receipts_per_day_x_receipts_per_day",
    # "log1p_receipts_per_day_x_trip_duration_days",
    "miles_per_day_x_receipts_per_day",
    # "log1p_miles_per_day_x_miles_per_receipt",
    # "log1p_receipts_per_day_x_miles_per_receipt",
    "miles_per_day_x_miles_per_receipt",
    "receipts_per_day_x_miles_per_receipt",
    "miles_per_day_x_receipts_per_mile",
    "receipts_per_day_x_receipts_per_mile",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()

    duration_safe = df_eng["trip_duration_days"].replace(0, 1)
    df_eng["miles_per_day"] = df_eng["miles_traveled"] / duration_safe
    df_eng["receipts_per_day"] = df_eng["total_receipts_amount"] / duration_safe
    if "expected_output" in df_eng.columns:
        df_eng["reimbursement_per_day"] = df_eng["expected_output"] / duration_safe
        df_eng["log1p_reimbursement_per_day"] = np.log1p(
            df_eng["reimbursement_per_day"]
        )
    # df_eng["log1p_miles_traveled"] = np.log1p(df_eng["miles_traveled"])
    # df_eng["log1p_trip_duration_days"] = np.log1p(df_eng["trip_duration_days"])
    # df_eng["log1p_total_receipts_amount"] = np.log1p(df_eng["total_receipts_amount"])
    # df_eng["log1p_miles_per_day"] = np.log1p(df_eng["miles_per_day"])
    # df_eng["log1p_receipts_per_day"] = np.log1p(df_eng["receipts_per_day"])
    # df_eng["miles_sq"] = df_eng["miles_traveled"] ** 2
    # df_eng["receipts_sq"] = df_eng["total_receipts_amount"] ** 2
    # df_eng["duration_sq"] = df_eng["trip_duration_days"] ** 2
    df_eng["miles_x_duration"] = df_eng["miles_traveled"] * df_eng["trip_duration_days"]
    df_eng["receipts_x_duration"] = (
        df_eng["total_receipts_amount"] * df_eng["trip_duration_days"]
    )
    df_eng["miles_x_receipts"] = (
        df_eng["miles_traveled"] * df_eng["total_receipts_amount"]
    )
    receipts_safe = df_eng["total_receipts_amount"].replace(0, 1)
    df_eng["miles_per_receipt"] = df_eng["miles_traveled"] / receipts_safe
    df_eng["receipts_per_mile"] = (
        df_eng["total_receipts_amount"] / df_eng["miles_traveled"]
    )

    main_features = [
        # "log1p_miles_per_day",
        # "log1p_receipts_per_day",
        "miles_per_day",
        "receipts_per_day",
        "miles_per_receipt",
        "receipts_per_mile",
    ]
    for i in range(len(main_features)):
        for j in range(i + 1, len(main_features)):
            f1 = main_features[i]
            f2 = main_features[j]
            col_name = f"{f1}_x_{f2}"
            df_eng[col_name] = df_eng[f1] * df_eng[f2]

    # log_features = [
    #     "log1p_miles_per_day",
    #     "log1p_receipts_per_day",
    # ]

    # for f in log_features:
    #     col_name = f"{f}_x_trip_duration_days"
    #     df_eng[col_name] = df_eng[f] * df_eng["trip_duration_days"]

    return df_eng


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        if hasattr(self.model, "fit"):
            try:
                self.model.fit(X, y, verbose=False)
            except TypeError:
                self.model.fit(X, y)
        else:
            raise TypeError("Provided model object does not have a fit method")

    def predict(self, X):
        # Ensure all feature columns are present
        for col in self.feature_names:
            if col not in X:
                X[col] = 0
        X = X[self.feature_names]
        return self.model.predict(X)


def load_and_train_model(
    model_factory, features, target_transform="raw", split_by_day=False
):
    df = pd.read_csv("public_cases.csv")
    df = engineer_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if target_transform == "log":
        df.dropna(subset=["log1p_reimbursement_per_day"], inplace=True)
        y = df["log1p_reimbursement_per_day"]
    else:
        y = df["expected_output"]

    X = df.loc[y.index, features]

    if not split_by_day:
        print(f"Fitting single model on '{target_transform}' target...")
        model_instance = model_factory()
        model = ModelWrapper(model_instance)
        model.fit(X, y)
        print("Model fitting complete.")
        return model, X.columns.tolist()
    else:
        print(f"Fitting daily models on '{target_transform}' target...")
        daily_models = {}
        unique_durations = X["trip_duration_days"].unique()

        for duration in sorted(unique_durations):
            duration_mask = X["trip_duration_days"] == duration
            if duration_mask.sum() < 10:
                print(
                    f"    Skipping duration {duration}, not enough samples ({duration_mask.sum()})."
                )
                continue

            print(
                f"  - Training model for duration: {int(duration)} days ({duration_mask.sum()} samples)"
            )

            X_duration = X[duration_mask]
            y_duration = y[duration_mask]

            model_instance = model_factory()
            model = ModelWrapper(model_instance)
            model.fit(X_duration, y_duration)
            daily_models[duration] = model

        print("Daily model fitting complete.")
        return daily_models, features


def calculate_reimbursement(
    model, feature_cols, target_transform="raw", split_by_day=False, **kwargs
):
    X = engineer_features(pd.DataFrame([kwargs]))

    if not split_by_day:
        pred = model.predict(X)[0]
    else:
        daily_models = model
        duration = kwargs["trip_duration_days"]

        if duration in daily_models:
            duration_model = daily_models[duration]
        else:
            # Fallback for unseen duration
            trained_durations = np.array(sorted(daily_models.keys()))
            closest_duration_idx = (np.abs(trained_durations - duration)).argmin()
            closest_duration = trained_durations[closest_duration_idx]
            print(
                f"Warning: No model for duration {duration}. Using model for closest duration: {int(closest_duration)} days."
            )
            duration_model = daily_models[closest_duration]

        pred = duration_model.predict(X)[0]

    if target_transform == "log":
        reimbursement_per_day = np.expm1(pred)
        duration = kwargs["trip_duration_days"]
        final_pred = reimbursement_per_day * duration
    else:
        final_pred = pred

    return round(float(final_pred), 2)
