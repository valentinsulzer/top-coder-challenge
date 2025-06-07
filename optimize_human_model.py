import numpy as np
import pandas as pd
from human_loop_model import HumanLoopModel


def main():
    """
    Trains the HumanLoopModel and saves its optimized parameters to a file.
    """
    print("Loading data...")
    df = pd.read_csv("public_cases.csv")

    # The same feature engineering from common_utils to create the target variable
    duration_safe = df["trip_duration_days"].replace(0, 1)
    if "expected_output" in df.columns:
        df["reimbursement_per_day"] = df["expected_output"] / duration_safe
        df["log1p_reimbursement_per_day"] = np.log1p(df["reimbursement_per_day"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log1p_reimbursement_per_day"], inplace=True)

    X = df
    y = df["log1p_reimbursement_per_day"]

    # Initialize and fit the HumanLoopModel
    model = HumanLoopModel()
    print("Starting optimization of human model parameters...")
    model.fit(X, y)

    # Save the optimized parameters
    params = model.params
    print(f"\nOptimized parameters: {params}")
    np.save("human_model_params.npy", params)
    print("✅ Parameters saved to human_model_params.npy")


if __name__ == "__main__":
    main()
