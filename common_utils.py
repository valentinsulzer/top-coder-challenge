import pandas as pd
import numpy as np
import os
from human_model import output_model


FEATURES = [
    "human_model_output",
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()

    # Add human model output as a feature if parameters are available
    params_path = "human_model_params.npy"
    if os.path.exists(params_path):
        human_params = np.load(params_path)
        df_eng["human_model_output"] = output_model(
            human_params,
            df_eng["miles_traveled"],
            df_eng["total_receipts_amount"],
            df_eng["trip_duration_days"],
        )
    else:
        # If the parameters don't exist, fill the column with zeros
        df_eng["human_model_output"] = 0

    duration_safe = df_eng["trip_duration_days"].replace(0, 1)
    df_eng["miles_per_day"] = df_eng["miles_traveled"] / duration_safe
    df_eng["receipts_per_day"] = df_eng["total_receipts_amount"] / duration_safe
    if "expected_output" in df_eng.columns:
        df_eng["reimbursement_per_day"] = df_eng["expected_output"] / duration_safe
        df_eng["log1p_reimbursement_per_day"] = np.log1p(
            df_eng["reimbursement_per_day"]
        )
    df_eng["log1p_miles_traveled"] = np.log1p(df_eng["miles_traveled"])
    df_eng["log1p_trip_duration_days"] = np.log1p(df_eng["trip_duration_days"])
    df_eng["log1p_total_receipts_amount"] = np.log1p(df_eng["total_receipts_amount"])
    df_eng["log1p_miles_per_day"] = np.log1p(df_eng["miles_per_day"])
    df_eng["log1p_receipts_per_day"] = np.log1p(df_eng["receipts_per_day"])
    df_eng["miles_sq"] = df_eng["miles_traveled"] ** 2
    df_eng["receipts_sq"] = df_eng["total_receipts_amount"] ** 2
    df_eng["duration_sq"] = df_eng["trip_duration_days"] ** 2
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
    for i in range(len(main_features)):
        for j in range(i + 1, len(main_features)):
            f1 = main_features[i]
            f2 = main_features[j]
            col_name = f"{f1}_x_{f2}"
            df_eng[col_name] = df_eng[f1] * df_eng[f2]
    return df_eng


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        # The fit method of the underlying model is called
        if hasattr(self.model, "fit"):
            # Handle models with a 'verbose' parameter in their fit method
            if "verbose" in self.model.get_params():
                self.model.fit(X, y, verbose=False)
            else:
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


def load_and_train_model(model_instance, features):
    df = pd.read_csv("public_cases.csv")
    df = engineer_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log1p_reimbursement_per_day"], inplace=True)
    X = df[features]
    y = df["log1p_reimbursement_per_day"]

    print("Fitting model...")
    model = ModelWrapper(model_instance)
    model.fit(X, y)
    print("Model fitting complete.")

    return model, X.columns.tolist()


def calculate_reimbursement(model, feature_cols, **kwargs):
    X = engineer_features(pd.DataFrame([kwargs]))
    for col in feature_cols:
        if col not in X:
            X[col] = 0
    X = X[feature_cols]
    log_pred = model.predict(X)[0]
    reimbursement_per_day = np.expm1(log_pred)
    duration = X["trip_duration_days"].iloc[0]
    total_reimbursement = reimbursement_per_day * duration
    return round(float(total_reimbursement), 2)
