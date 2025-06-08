import argparse
from common_utils import FEATURES
from models import MODEL_CONFIGS


def get_model_factory_and_name(model_key="xgboost"):
    """Gets the model factory and name from the config."""
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        raise ValueError(f"Model '{model_key}' not found in MODEL_CONFIGS.")
    return config["model"], config["name"]


def parse_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run the reimbursement prediction model."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="xgboost",
        choices=MODEL_CONFIGS.keys(),
        help="Model to use.",
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        default="raw",
        choices=["raw", "log"],
        help="Target transformation to apply ('raw' or 'log').",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=FEATURES,
        help=f"List of features to use for training. Defaults to: {', '.join(FEATURES)}",
    )
    parser.add_argument(
        "--split-by-day",
        action="store_true",
        help="Train a separate model for each trip duration.",
    )
    parser.add_argument(
        "--use-day-buckets",
        action="store_true",
        help="Train a separate model for day buckets (e.g., 1-2, 3-5, 6-10, 11+).",
    )
    parser.add_argument(
        "--train-test-split",
        action="store_true",
        help="Run a train-test split for each model to evaluate overfitting.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining the model even if a saved version exists.",
    )

    return parser.parse_args()
