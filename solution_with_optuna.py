import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from decimal import Decimal
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error


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


class ReimbursementModel:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()

        def objective(trial):
            params = {
                "objective": "regression_l1",
                "metric": "mae",
                "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
                "boosting_type": "gbdt",
            }

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        self.model = lgb.LGBMRegressor(**trial.params)
        self.model.fit(X, y)

    def predict(self, X):
        for col in self.feature_names:
            if col not in X:
                X[col] = 0
        X = X[self.feature_names]
        return self.model.predict(X)


def _load_model(features):
    df = pd.read_csv("public_cases.csv")
    df = engineer_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log1p_reimbursement_per_day"], inplace=True)
    X = df[features]
    y = df["log1p_reimbursement_per_day"]
    model = ReimbursementModel()
    model.fit(X, y)
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

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Evaluation mode
        print("🧾 Black Box Challenge - Reimbursement System Evaluation")
        print("=======================================================")
        print()
        csv_path = "public_cases.csv"
        if not os.path.exists(csv_path):
            print(f"❌ Error: {csv_path} not found!")
            sys.exit(1)
        print("📊 Running evaluation against 1,000 test cases...")
        print("Fitting model once...")
        model, feature_cols = _load_model(FEATURES)
        print("Model fitting complete.")
        print()
        df = pd.read_csv(csv_path)
        num_cases = len(df)
        successful_runs = 0
        exact_matches = 0
        close_matches = 0
        total_error = Decimal("0")
        max_error = Decimal("0")
        results = []
        errors = []
        for i, row in df.iterrows():
            if i % 100 == 0:
                print(f"Progress: {i}/{num_cases} cases processed...", file=sys.stderr)
            try:
                expected = Decimal(str(row["expected_output"]))
                actual = Decimal(
                    str(
                        calculate_reimbursement(
                            model,
                            feature_cols,
                            trip_duration_days=row["trip_duration_days"],
                            miles_traveled=row["miles_traveled"],
                            total_receipts_amount=row["total_receipts_amount"],
                        )
                    )
                )
                error = abs(actual - expected)
                results.append(
                    {
                        "case_num": i + 1,
                        "expected": expected,
                        "actual": actual,
                        "error": error,
                        "trip_duration": row["trip_duration_days"],
                        "miles_traveled": row["miles_traveled"],
                        "receipts_amount": row["total_receipts_amount"],
                    }
                )
                successful_runs += 1
                if error < Decimal("0.01"):
                    exact_matches += 1
                if error < Decimal("1.00"):
                    close_matches += 1
                total_error += error
                if error > max_error:
                    max_error = error
            except Exception as e:
                errors.append(f"Case {i + 1}: Calculation failed with error: {e}")
        if successful_runs == 0:
            print("❌ No successful test cases!")
            print("Check logs for errors.")
        else:
            avg_error = total_error / successful_runs
            exact_pct = (Decimal(exact_matches) / successful_runs) * 100
            close_pct = (Decimal(close_matches) / successful_runs) * 100
            print("✅ Evaluation Complete!")
            print("\n📈 Results Summary:")
            print(f"  Total test cases: {num_cases}")
            print(f"  Successful runs: {successful_runs}")
            print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
            print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
            print(f"  Average error: ${avg_error:.2f}")
            print(f"  Maximum error: ${max_error:.2f}")
            score = avg_error * 100 + (num_cases - exact_matches) * Decimal("0.1")
            print(f"\n🎯 Your Score: {score:.2f} (lower is better)")

    # Final Evaluation Results:
    # 🧾 Black Box Challenge - Reimbursement System Evaluation
    # =======================================================
    #
    # 📊 Running evaluation against 1,000 test cases...
    # Fitting model once...
    # Model fitting complete.
    #
    # ✅ Evaluation Complete!
    #
    # 📈 Results Summary:
    #   Total test cases: 1000
    #   Successful runs: 1000
    #   Exact matches (±$0.01): 13 (1.3%)
    #   Close matches (±$1.00): 294 (29.4%)
    #   Average error: $36.66
    #   Maximum error: $1000.33
    #
    # 🎯 Your Score: 3764.93 (lower is better)

    elif len(sys.argv) == 4:
        # Prediction mode
        try:
            trip_duration_days = int(sys.argv[1])
            miles_traveled = float(sys.argv[2])
            total_receipts_amount = float(sys.argv[3])
        except ValueError:
            print(
                "Error: Invalid input types. Please provide integers for days and miles, and a float for receipts."
            )
            sys.exit(1)
        model, feature_cols = _load_model(FEATURES)
        reimbursement_amount = calculate_reimbursement(
            model,
            feature_cols,
            trip_duration_days=trip_duration_days,
            miles_traveled=miles_traveled,
            total_receipts_amount=total_receipts_amount,
        )
        print(reimbursement_amount)
    else:
        print(
            "Usage: python solution.py [<trip_duration_days> <miles_traveled> <total_receipts_amount>]"
        )
        sys.exit(1)
