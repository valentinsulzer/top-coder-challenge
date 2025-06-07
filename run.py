import sys
import argparse
from common_utils import (
    FEATURES,
    load_and_train_model,
    calculate_reimbursement,
)
from evaluation_runner import run_evaluation
from models import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        description="Run the Reimbursement Prediction Challenge."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="lightgbm",
        help="The model to run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluate", "predict"],
        default="evaluate",
        help="The mode to run the script in.",
    )
    parser.add_argument(
        "--days", type=int, help="Trip duration in days for prediction."
    )
    parser.add_argument("--miles", type=float, help="Miles traveled for prediction.")
    parser.add_argument("--receipts", type=float, help="Total receipts for prediction.")

    args = parser.parse_args()

    model_key = args.model
    if model_key != "all":
        model_config = MODEL_CONFIGS[model_key]
        model_instance = model_config["model"]()
        model_name = model_config["name"]

    if args.mode == "evaluate":
        if model_key == "all":
            for model_name, config in MODEL_CONFIGS.items():
                print(f"--- Running evaluation for {config['name']} ---")
                model_instance = config["model"]()
                model, feature_cols = load_and_train_model(model_instance, FEATURES)
                run_evaluation(model, config["name"], feature_cols)
                print(f"--- Finished evaluation for {config['name']} ---\n")
        else:
            model_instance = MODEL_CONFIGS[model_key]["model"]()
            model, feature_cols = load_and_train_model(model_instance, FEATURES)
            run_evaluation(model, MODEL_CONFIGS[model_key]["name"], feature_cols)
    elif args.mode == "predict":
        if not all([args.days, args.miles, args.receipts]):
            print(
                "Error: For prediction mode, you must provide --days, --miles, and --receipts."
            )
            sys.exit(1)
        model, feature_cols = load_and_train_model(model_instance, FEATURES)
        reimbursement = calculate_reimbursement(
            model,
            feature_cols,
            trip_duration_days=args.days,
            miles_traveled=args.miles,
            total_receipts_amount=args.receipts,
        )
        print(f"Predicted Reimbursement: ${reimbursement}")


if __name__ == "__main__":
    main()
