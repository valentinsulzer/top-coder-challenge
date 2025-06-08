import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from common_utils import engineer_features
import shap


def run_evaluation(model, model_name, feature_cols, target_transform="raw"):
    """
    Runs a full evaluation for a given model.

    Args:
        model: The trained model object.
        model_name (str): The name of the model for titles and filenames.
        feature_cols (list): The list of feature names.
        target_transform (str): The transformation to apply to the model's predictions.
    """
    print(f"🧾 Black Box Challenge - Reimbursement System Evaluation ({model_name})")
    print("===================================================================")
    print()

    output_suffix = model_name.lower().replace(" ", "_")
    output_dir = f"visualizations_{output_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # Feature Importance Plot (if available)
    if hasattr(model.model, "feature_importances_"):
        print("📊 Generating feature importance plot...")
        feature_importances = model.model.feature_importances_
        importance_df = (
            pd.DataFrame({"feature": feature_cols, "importance": feature_importances})
            .sort_values("importance", ascending=False)
            .head(20)
        )

        plt.figure(figsize=(12, 10))
        sns.barplot(x="importance", y="feature", data=importance_df)
        plt.title(f"Top 20 Feature Importances ({model_name})")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Feature importance plot saved to '{plot_path}'")
        print()
    else:
        print(f"📊 Feature importance plot not available for {model_name}.")

    # Predictions and Results Saving
    df = pd.read_csv("public_cases.csv")
    print("⚙️  Engineering features for evaluation set...")
    df_eval = engineer_features(df)

    # SHAP Analysis for tree-based models
    if hasattr(model.model, "predict") and hasattr(model.model, "feature_importances_"):
        print("📊 Generating SHAP summary plot...")
        # Using a subset of data for performance reasons
        X_sampled = df_eval[feature_cols].sample(100, random_state=42)
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X_sampled)

        plt.figure()
        shap.summary_plot(shap_values, X_sampled, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance ({model_name})")
        plt.tight_layout()
        shap_plot_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"  - SHAP summary plot saved to '{shap_plot_path}'")
        print()

    if "xgboost" in model_name.lower():
        print("🔍 Analyzing XGBoost model structure...")
        booster = model.model.get_booster()

        model_dump = booster.get_dump(dump_format="text")

        split_pattern = re.compile(r"\[(.*?)<([^\]]+)\]")

        splits = []
        for tree in model_dump:
            for line in tree.split("\n"):
                match = split_pattern.search(line)
                if match:
                    feature_name = match.group(1)
                    threshold = float(match.group(2))

                    # Simplified extraction, gain is not easily available in text dump with feature names
                    splits.append(
                        {
                            "feature": feature_name,
                            "threshold": threshold,
                        }
                    )

        if splits:
            splits_df = pd.DataFrame(splits)

            # Show top 10 features by split frequency
            print("\n📊 Top 10 Features by Split Frequency:")
            feature_split_counts = splits_df["feature"].value_counts().head(10)
            print(feature_split_counts.to_string())

            # Show most common thresholds for top features
            print("\n\n📋 Common Thresholds for Top Features:")
            for feature in feature_split_counts.index:
                print(f"\n  - Feature: {feature}")
                thresholds = splits_df[splits_df["feature"] == feature]["threshold"]
                if len(thresholds.unique()) > 1:
                    try:
                        quantiles = pd.qcut(
                            thresholds,
                            q=min(5, len(thresholds.unique()) - 1),
                            duplicates="drop",
                        )
                        print("    Common threshold ranges (quantiles):")
                        print(
                            quantiles.value_counts(normalize=True)
                            .sort_index()
                            .to_string()
                        )
                    except ValueError:
                        print("    Could not determine interesting threshold ranges.")
                else:
                    print(f"    Single threshold: {thresholds.iloc[0]}")
            print()
        else:
            print("  - No splits found in the model dump.")
        print()

    print("🔮 Predicting on evaluation set...")
    preds = model.predict(df_eval)

    if target_transform == "log":
        reimbursement_per_day = np.expm1(preds)
        duration = df_eval["trip_duration_days"]
        predicted_output = reimbursement_per_day * duration
    else:
        predicted_output = preds

    df_eval["predicted_output"] = np.round(predicted_output, 2)
    df_eval["error"] = abs(df_eval["predicted_output"] - df_eval["expected_output"])

    output_csv_path = f"evaluation_results_with_features_{output_suffix}.csv"
    print(f"💾 Saving full evaluation results to '{output_csv_path}'...")
    df_eval.to_csv(output_csv_path, index=False)
    print(f"  - Saved successfully.")
    print()

    # Metrics Calculation
    print("📝 Calculating metrics...")
    error = df_eval["error"]

    num_cases = len(df_eval)
    successful_runs = len(df_eval)
    exact_matches = (error < 0.01).sum()
    close_matches = (error < 1.00).sum()
    total_error = error.sum()
    max_error = error.max()

    if successful_runs == 0:
        print("❌ No successful test cases!")
    else:
        avg_error = total_error / successful_runs
        exact_pct = (exact_matches / successful_runs) * 100
        close_pct = (close_matches / successful_runs) * 100
        print("✅ Evaluation Complete!")
        print(f"\n📈 Results Summary ({model_name}):")
        print(f"  Total test cases: {num_cases}")
        print(f"  Successful runs: {successful_runs}")
        print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
        print(f"  Average error: ${avg_error:.2f}")
        print(f"  Maximum error: ${max_error:.2f}")

        # Visualizations
        print("\n📊 Generating evaluation visualizations...")
        results_df = pd.DataFrame(
            {
                "expected": df_eval["expected_output"],
                "actual": df_eval["predicted_output"],
                "error": error,
                "case_num": df_eval.index + 1,
                "trip_duration": df_eval["trip_duration_days"],
                "miles_traveled": df_eval["miles_traveled"],
                "receipts_amount": df_eval["total_receipts_amount"],
            }
        )

        # Scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="expected", y="actual", data=results_df, alpha=0.5)
        plt.title(f"Expected vs. Actual Reimbursement ({model_name})")
        plt.xlabel("Expected Reimbursement ($)")
        plt.ylabel("Actual Reimbursement ($)")
        max_val = max(results_df["expected"].max(), results_df["actual"].max())
        min_val = min(results_df["expected"].min(), results_df["actual"].min())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, "predictions_scatter.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Scatter plot of predictions saved to '{plot_path}'")

        # Error distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["error"], bins=50, kde=True)
        plt.title(f"Distribution of Prediction Errors ({model_name})")
        plt.xlabel("Absolute Prediction Error ($)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plot_path = os.path.join(output_dir, "error_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Error distribution plot saved to '{plot_path}'")

        # Worst predictions
        print("\n📋 Top 10 Worst Predictions (Largest Error):")
        worst_predictions = results_df.sort_values("error", ascending=False).head(10)
        print(worst_predictions.to_string())

        score = avg_error * 100 + (num_cases - exact_matches) * 0.1
        print(f"\n🎯 Your Score: {score:.2f} (lower is better)")
