import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from common_utils import engineer_features, FEATURES


def analyze_discrepancies():
    """
    Analyzes the data to find cases where similar inputs lead to different outputs.
    """
    # Load and prepare the data
    df = pd.read_csv("public_cases.csv")
    df = engineer_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["log1p_reimbursement_per_day"], inplace=True)

    # Separate features and target
    X = df[FEATURES].copy()
    y = df["log1p_reimbursement_per_day"].copy()

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate pairwise distances
    input_distances = pairwise_distances(X_scaled, metric="euclidean")
    output_differences = np.abs(y.values[:, np.newaxis] - y.values)

    # Define thresholds for discrepancy
    # A small input distance is below the 20th percentile of all input distances
    input_dist_threshold = np.percentile(
        input_distances[np.triu_indices_from(input_distances, k=1)], 20
    )
    # A large output difference is above the 80th percentile of all output differences
    output_diff_threshold = np.percentile(
        output_differences[np.triu_indices_from(output_differences, k=1)], 80
    )

    print(f"Input distance threshold (20th percentile): {input_dist_threshold:.4f}")
    print(f"Output difference threshold (80th percentile): {output_diff_threshold:.4f}")
    print("\\nFinding discrepant cases...\\n")

    # Find and save discrepant cases
    discrepant_cases = []

    # Get the upper triangle indices to avoid duplicate pairs and self-comparisons
    upper_triangle_indices = np.triu_indices_from(input_distances, k=1)

    for i, j in zip(*upper_triangle_indices):
        if (
            input_distances[i, j] < input_dist_threshold
            and output_differences[i, j] > output_diff_threshold
        ):
            discrepant_cases.append(
                {
                    "case_1_index": df.index[i],
                    "case_2_index": df.index[j],
                    "input_distance": input_distances[i, j],
                    "output_difference": output_differences[i, j],
                    "case_1_trip_duration_days": df.iloc[i]["trip_duration_days"],
                    "case_1_miles_traveled": df.iloc[i]["miles_traveled"],
                    "case_1_total_receipts_amount": df.iloc[i]["total_receipts_amount"],
                    "case_1_actual_reimbursement": np.expm1(
                        df.iloc[i]["log1p_reimbursement_per_day"]
                    )
                    * df.iloc[i]["trip_duration_days"],
                    "case_2_trip_duration_days": df.iloc[j]["trip_duration_days"],
                    "case_2_miles_traveled": df.iloc[j]["miles_traveled"],
                    "case_2_total_receipts_amount": df.iloc[j]["total_receipts_amount"],
                    "case_2_actual_reimbursement": np.expm1(
                        df.iloc[j]["log1p_reimbursement_per_day"]
                    )
                    * df.iloc[j]["trip_duration_days"],
                }
            )

    output_filename = "discrepancies.csv"
    if not discrepant_cases:
        print("No significant discrepancies found with the current thresholds.")
    else:
        discrepancies_df = pd.DataFrame(discrepant_cases)
        discrepancies_df.to_csv(output_filename, index=False, float_format="%.4f")
        print(f"Found {len(discrepancies_df)} discrepant pairs.")
        print(f"Discrepancy analysis saved to {output_filename}")


if __name__ == "__main__":
    analyze_discrepancies()
