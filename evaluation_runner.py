import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from common_utils import (
    load_and_train_model,
    _get_day_bucket,
    engineer_features,
)
from models import MODEL_CONFIGS
from run import parse_args, get_model_factory_and_name
import warnings


def analyze_model_splits(model, feature_names, model_name, n_splits=10):
    print("Model dump could not be parsed.")


def analyze_feature_importance_shap(model, X, feature_names, model_name):
    """Analyze feature importance using SHAP."""
    print(f"\n--- SHAP Feature Importance Analysis for {model_name} ---")

    if not hasattr(model, "predict"):
        print("Model does not support SHAP analysis (no predict method).")
        return

    # Using a subset of data for performance reasons
    X_sampled = X.sample(min(100, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sampled)

    plt.figure()
    shap.summary_plot(
        shap_values, X_sampled, feature_names=feature_names, show=False, plot_type="bar"
    )
    plt.title(f"SHAP Feature Importance for {model_name}")
    plt.tight_layout()
    plt.savefig(f"shap_summary_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    print("SHAP summary plot saved.")


def run_evaluation():
    args = parse_args()
    model_factory, model_name = get_model_factory_and_name(args.model)

    model, feature_cols, split_scores = load_and_train_model(
        model_factory,
        features=args.features,
        target_transform=args.target_transform,
        split_by_day=args.split_by_day,
        train_test_split=args.train_test_split,
        use_day_buckets=args.use_day_buckets,
        force_retrain=args.force_retrain,
    )

    if isinstance(model, dict):
        # Multi-model scenario (daily or bucketed)
        print("\n--- Evaluating Multi-Model Performance ---")

        if split_scores:
            all_test_maes = [
                scores["test_mae"] for scores in split_scores.values() if scores
            ]
            if all_test_maes:
                avg_test_mae = np.mean(all_test_maes)
                print(f"\nAverage Test MAE across all groups: ${avg_test_mae:.2f}")

        # Cannot do single SHAP analysis for multiple models easily
        print("\nSkipping model analysis for multi-model setup.")

    else:
        # Single model scenario
        print(f"\n--- Analyzing Single {model_name} Model ---")

        # Create a dummy X from training data for SHAP analysis
        df = pd.read_csv("public_cases.csv")
        X_for_shap = df[feature_cols]

        analyze_feature_importance_shap(model, X_for_shap, feature_cols, model_name)
        if hasattr(model.model, "get_booster"):
            analyze_model_splits(model.model, feature_cols, model_name)

    # ... evaluation logic on test cases ...
    try:
        df_test = pd.read_csv("private_cases.csv")
        df_test = engineer_features(df_test)

        print("\n--- Predicting on Private Test Set ---")
        if not args.split_by_day and not args.use_day_buckets:
            preds = model.predict(df_test)
        else:
            models = model
            df_test["preds"] = np.nan

            if args.use_day_buckets:
                df_test["day_bucket"] = df_test["trip_duration_days"].apply(
                    _get_day_bucket
                )
                iterator = df_test["day_bucket"].unique()
                id_col = "day_bucket"
            else:  # split_by_day
                iterator = df_test["trip_duration_days"].unique()
                id_col = "trip_duration_days"

            trained_keys = np.array(sorted(models.keys()))

            for value in iterator:
                mask = df_test[id_col] == value
                X_subset = df_test[mask]

                if value in models:
                    current_model = models[value]
                else:
                    # Fallback for unseen duration/bucket
                    # Find the closest key (numerically for days, or just first for buckets)
                    if id_col == "trip_duration_days":
                        # find closest day
                        closest_key_index = np.abs(
                            trained_keys.astype(int) - int(value)
                        ).argmin()
                        closest_key = trained_keys[closest_key_index]
                    else:  # buckets
                        closest_key = trained_keys[0]  # simple fallback for now

                    print(
                        f"Warning: No model for {id_col} {value}. Using model for closest key: {closest_key}."
                    )
                    current_model = models[closest_key]

                pred_subset = current_model.predict(
                    X_subset.drop(columns=["day_bucket"], errors="ignore")
                )
                df_test.loc[mask, "preds"] = pred_subset

            preds = df_test["preds"].values

        if args.target_transform == "log":
            reimbursement_per_day = np.expm1(preds)
            duration = df_test["trip_duration_days"]
            final_preds = reimbursement_per_day * duration
        else:
            final_preds = preds

        df_results = pd.DataFrame(
            {
                "case_id": df_test["case_id"],
                "predicted_reimbursement": np.round(final_preds, 2),
            }
        )

        if "expected_output" in df_test.columns:
            df_results["expected_output"] = df_test["expected_output"]
            df_results["absolute_error"] = abs(
                df_results["predicted_reimbursement"] - df_results["expected_output"]
            )

        df_results.to_csv("submission.csv", index=False)
        print("\nSubmission file 'submission.csv' created.")

        if "absolute_error" in df_results.columns:
            final_mae = df_results["absolute_error"].mean()
            print(f"\nFinal MAE on private test set: ${final_mae:.2f}")
    except FileNotFoundError:
        print(
            "\n'private_cases.csv' not found. Skipping final evaluation against private data."
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run_evaluation()
