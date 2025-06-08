import argparse
from common_utils import (
    FEATURES,
    load_and_train_model,
    calculate_reimbursement,
)
from run import get_model_factory_and_name


def main():
    """
    Main function to predict reimbursement for a single case using a pre-trained model.
    """
    parser = argparse.ArgumentParser(
        description="Predict reimbursement for a single travel case."
    )
    parser.add_argument("days", type=int, help="Trip duration in days for prediction.")
    parser.add_argument("miles", type=float, help="Miles traveled for prediction.")
    parser.add_argument(
        "receipts", type=float, help="Total receipts amount for prediction."
    )
    args = parser.parse_args()

    # Load the model factory for our best model
    model_factory, _ = get_model_factory_and_name("xgboost")

    # Load the pre-trained models from disk, suppressing status messages
    model, feature_cols, _ = load_and_train_model(
        model_factory,
        features=FEATURES,
        use_day_buckets=True,  # Use our best strategy
        verbose=False,  # Suppress prints to keep stdout clean
    )

    # Calculate the reimbursement for the given inputs
    reimbursement = calculate_reimbursement(
        model,
        feature_cols=feature_cols,
        use_day_buckets=True,
        verbose=False,
        # Pass case data
        trip_duration_days=args.days,
        miles_traveled=args.miles,
        total_receipts_amount=args.receipts,
    )

    # Print the final reimbursement amount, which is expected by eval.sh
    print(reimbursement)


if __name__ == "__main__":
    main()
