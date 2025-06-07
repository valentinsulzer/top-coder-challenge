import pandas as pd
import numpy as np


def summarize_discrepancies():
    """
    Reads the discrepancies.csv file and generates a summary of the findings.
    """
    try:
        df = pd.read_csv("discrepancies.csv")
    except FileNotFoundError:
        print("discrepancies.csv not found. Please run analyze_discrepancies.py first.")
        return

    output_filename = "discrepancy_summary.txt"
    with open(output_filename, "w") as f:
        f.write("Discrepancy Analysis Summary\n")
        f.write("=" * 30 + "\n\n")

        f.write(f"Total number of discrepant pairs found: {len(df)}\n\n")

        f.write("Summary Statistics for Distances and Differences:\n")
        f.write(df[["input_distance", "output_difference"]].describe().to_string())
        f.write("\n\n")

        f.write("Summary Statistics for Case 1 Inputs and Outputs:\n")
        case1_cols = [c for c in df.columns if "case_1" in c and "index" not in c]
        f.write(df[case1_cols].describe().to_string())
        f.write("\n\n")

        f.write("Summary Statistics for Case 2 Inputs and Outputs:\n")
        case2_cols = [c for c in df.columns if "case_2" in c and "index" not in c]
        f.write(df[case2_cols].describe().to_string())
        f.write("\n\n")

        # Find most frequent cases in discrepancies
        case_1_counts = df["case_1_index"].value_counts()
        case_2_counts = df["case_2_index"].value_counts()
        total_counts = case_1_counts.add(case_2_counts, fill_value=0).sort_values(
            ascending=False
        )

        f.write("Top 10 Most Frequent Cases in Discrepancies:\n")
        f.write(total_counts.head(10).to_string())
        f.write("\n\n")

    print(f"Discrepancy summary saved to {output_filename}")


if __name__ == "__main__":
    summarize_discrepancies()
