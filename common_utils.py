import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "trip_duration_days",
    "miles_traveled",
    "total_receipts_amount",
    "miles_per_day",
    "receipts_per_day",
    "miles_x_duration",
    "receipts_x_duration",
    "miles_x_receipts",
    "miles_per_receipt",
    "receipts_per_mile",
]


def _get_day_bucket(duration):
    if duration <= 2:
        return "1-2"
    elif duration <= 5:
        return "3-5"
    elif duration <= 10:
        return "6-10"
    else:
        return "11+"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    duration_safe = df_eng["trip_duration_days"].replace(0, epsilon)
    receipts_safe = df_eng["total_receipts_amount"].replace(0, epsilon)
    miles_safe = df_eng["miles_traveled"].replace(0, epsilon)

    df_eng["miles_per_day"] = df_eng["miles_traveled"] / duration_safe
    df_eng["receipts_per_day"] = df_eng["total_receipts_amount"] / duration_safe
    if "expected_output" in df_eng.columns:
        df_eng["reimbursement_per_day"] = df_eng["expected_output"] / duration_safe
        df_eng["log1p_reimbursement_per_day"] = np.log1p(
            df_eng["reimbursement_per_day"]
        )
    df_eng["miles_x_duration"] = df_eng["miles_traveled"] * df_eng["trip_duration_days"]
    df_eng["receipts_x_duration"] = (
        df_eng["total_receipts_amount"] * df_eng["trip_duration_days"]
    )
    df_eng["miles_x_receipts"] = (
        df_eng["miles_traveled"] * df_eng["total_receipts_amount"]
    )
    df_eng["miles_per_receipt"] = df_eng["miles_traveled"] / receipts_safe
    df_eng["receipts_per_mile"] = df_eng["total_receipts_amount"] / miles_safe

    # Clip extreme values to prevent overflow
    for col in df_eng.columns:
        if df_eng[col].dtype in [np.float64, np.int64]:
            df_eng[col] = np.clip(
                df_eng[col], np.finfo(np.float32).min, np.finfo(np.float32).max
            )

    return df_eng


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.feature_names = None
        self.scaler = None

    def fit(self, X, y, eval_set=None):
        self.feature_names = X.columns.tolist()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if eval_set:
            X_eval, y_eval = eval_set
            X_eval_scaled = self.scaler.transform(X_eval)
            fit_params = {
                "eval_set": [(X_eval_scaled, y_eval)],
                "early_stopping_rounds": 50,
                "verbose": False,
            }
        else:
            fit_params = {}

        if hasattr(self.model, "fit"):
            try:
                self.model.fit(X_scaled, y, **fit_params)
            except TypeError:
                # Fallback for models not supporting early stopping
                if "eval_set" in fit_params:
                    del fit_params["eval_set"]
                    del fit_params["early_stopping_rounds"]
                    del fit_params["verbose"]
                self.model.fit(X_scaled, y, **fit_params)
        else:
            raise TypeError("Provided model object does not have a fit method")

    def predict(self, X):
        # Ensure all feature columns are present
        for col in self.feature_names:
            if col not in X:
                X[col] = 0
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def _run_train_test_split(model_factory, X, y, target_transform, test_size=0.2):
    """Helper to run a single train-test split and return scores."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model_instance = model_factory()
    model = ModelWrapper(model_instance)
    # Use the test set as the eval set for early stopping
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    y_train_raw, y_test_raw = y_train, y_test
    if target_transform == "log":
        train_preds = np.expm1(train_preds)
        y_train_raw = np.expm1(y_train)
        test_preds = np.expm1(test_preds)
        y_test_raw = np.expm1(y_test)

    return {
        "train_mae": mean_absolute_error(y_train_raw, train_preds),
        "test_mae": mean_absolute_error(y_test_raw, test_preds),
    }


def load_and_train_model(
    model_factory,
    features,
    target_transform="raw",
    split_by_day=False,
    train_test_split=False,
    use_day_buckets=False,
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
    split_scores = None

    if train_test_split:
        print("Running train-test split...")
        if not split_by_day and not use_day_buckets:
            split_scores = _run_train_test_split(model_factory, X, y, target_transform)
            print(
                f"  - Single Model -> Train MAE: ${split_scores['train_mae']:.2f}, Test MAE: ${split_scores['test_mae']:.2f}"
            )
        else:
            daily_split_scores = {}
            if use_day_buckets:
                df["day_bucket"] = df["trip_duration_days"].apply(_get_day_bucket)
                X["day_bucket"] = df["day_bucket"]
                iterator = df["day_bucket"].unique()
                id_col = "day_bucket"
                print("  Splitting by day buckets...")
            else:  # split_by_day
                iterator = X["trip_duration_days"].unique()
                id_col = "trip_duration_days"
                print("  Splitting by individual days...")

            for value in sorted(iterator):
                mask = X[id_col] == value
                if mask.sum() < 10:
                    continue

                X_subset = X[mask].drop(columns=["day_bucket"], errors="ignore")
                y_subset = y[mask]

                scores = _run_train_test_split(
                    model_factory, X_subset, y_subset, target_transform
                )
                daily_split_scores[value] = scores

                if use_day_buckets:
                    print(
                        f"    - Bucket: {str(value).ljust(5)} | Train MAE: ${scores['train_mae']:.2f}, Test MAE: ${scores['test_mae']:.2f}"
                    )
                else:
                    print(
                        f"    - Duration: {int(value):>2} days | Train MAE: ${scores['train_mae']:<7.2f} | Test MAE: ${scores['test_mae']:<7.2f}"
                    )
            split_scores = daily_split_scores
        print()

    # Train final model(s) on all data
    if not split_by_day and not use_day_buckets:
        print(f"Fitting single model on '{target_transform}' target...")
        model_instance = model_factory()
        model = ModelWrapper(model_instance)
        model.fit(X, y)
        print("Model fitting complete.")
        return model, X.columns.tolist(), split_scores
    else:
        if use_day_buckets:
            df["day_bucket"] = df["trip_duration_days"].apply(_get_day_bucket)
            X["day_bucket"] = df["day_bucket"]
            iterator = df["day_bucket"].unique()
            id_col = "day_bucket"
            print(f"Fitting models for day buckets on '{target_transform}' target...")
        else:  # split_by_day
            iterator = X["trip_duration_days"].unique()
            id_col = "trip_duration_days"
            print(f"Fitting daily models on '{target_transform}' target...")

        models = {}
        for value in sorted(iterator):
            mask = X[id_col] == value
            if mask.sum() < 10:
                print(
                    f"    Skipping {id_col} {value}, not enough samples ({mask.sum()})."
                )
                continue

            print(f"  - Training model for {id_col}: {value} ({mask.sum()} samples)")

            X_subset = X[mask].drop(columns=["day_bucket"] if use_day_buckets else [])
            y_subset = y[mask]

            model_instance = model_factory()
            model = ModelWrapper(model_instance)
            model.fit(X_subset, y_subset)
            models[value] = model

        print("Model fitting complete.")
        return models, features, split_scores


def calculate_reimbursement(
    model,
    feature_cols,
    target_transform="raw",
    split_by_day=False,
    use_day_buckets=False,
    **kwargs,
):
    X = engineer_features(pd.DataFrame([kwargs]))

    if not split_by_day and not use_day_buckets:
        pred = model.predict(X)[0]
    else:
        models = model
        duration = kwargs["trip_duration_days"]

        if use_day_buckets:
            key = _get_day_bucket(duration)
            msg_type = "bucket"
        else:
            key = duration
            msg_type = "duration"

        if key in models:
            duration_model = models[key]
        else:
            # Fallback for unseen duration/bucket
            trained_keys = np.array(sorted(models.keys()))
            # This fallback is simpler for buckets; might need refinement
            closest_key = trained_keys[0]
            print(
                f"Warning: No model for {msg_type} {key}. Using model for closest {msg_type}: {closest_key}."
            )
            duration_model = models[closest_key]

        pred = duration_model.predict(X.drop(columns=["day_bucket"], errors="ignore"))[
            0
        ]

    if target_transform == "log":
        reimbursement_per_day = np.expm1(pred)
        duration = kwargs["trip_duration_days"]
        final_pred = reimbursement_per_day * duration
    else:
        final_pred = pred

    return round(float(final_pred), 2)
