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
        "--target-transform",
        type=str,
        choices=["log", "raw"],
        default="raw",
        help="Transformation to apply to the target variable before training.",
    )
    parser.add_argument(
        "--split-by-day",
        action="store_true",
        help="Train a separate model for each trip duration.",
    )
    parser.add_argument(
        "--train-test-split",
        action="store_true",
        help="Perform a single train-test split and report scores.",
    )
    parser.add_argument(
        "--day-buckets",
        action="store_true",
        help="Train a separate model for buckets of trip durations.",
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
        model_factory = model_config["model"]
        model_name = model_config["name"]

    if args.mode == "evaluate":
        if model_key == "all":
            for model_name, config in MODEL_CONFIGS.items():
                print(f"--- Running evaluation for {config['name']} ---")
                model_factory = config["model"]
                model, feature_cols, split_scores = load_and_train_model(
                    model_factory,
                    FEATURES,
                    target_transform=args.target_transform,
                    split_by_day=args.split_by_day,
                    train_test_split=args.train_test_split,
                    use_day_buckets=args.day_buckets,
                )
                run_evaluation(
                    model,
                    config["name"],
                    feature_cols,
                    target_transform=args.target_transform,
                    split_by_day=args.split_by_day,
                    split_scores=split_scores,
                    use_day_buckets=args.day_buckets,
                )
                print(f"--- Finished evaluation for {config['name']} ---\n")
        else:
            model, feature_cols, split_scores = load_and_train_model(
                model_factory,
                FEATURES,
                target_transform=args.target_transform,
                split_by_day=args.split_by_day,
                train_test_split=args.train_test_split,
                use_day_buckets=args.day_buckets,
            )
            run_evaluation(
                model,
                MODEL_CONFIGS[model_key]["name"],
                feature_cols,
                target_transform=args.target_transform,
                split_by_day=args.split_by_day,
                split_scores=split_scores,
                use_day_buckets=args.day_buckets,
            )
    elif args.mode == "predict":
        if not all([args.days, args.miles, args.receipts]):
            print(
                "Error: For prediction mode, you must provide --days, --miles, and --receipts."
            )
            sys.exit(1)
        model, feature_cols, _ = load_and_train_model(
            model_factory,
            FEATURES,
            target_transform=args.target_transform,
            split_by_day=args.split_by_day,
            train_test_split=args.train_test_split,
            use_day_buckets=args.day_buckets,
        )
        reimbursement = calculate_reimbursement(
            model,
            feature_cols,
            trip_duration_days=args.days,
            miles_traveled=args.miles,
            total_receipts_amount=args.receipts,
            target_transform=args.target_transform,
            split_by_day=args.split_by_day,
            use_day_buckets=args.day_buckets,
        )
        print(f"Predicted Reimbursement: ${reimbursement}")


if __name__ == "__main__":
    main()
