import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_discrepancies():
    """
    Reads the discrepancies.csv file and generates histograms for the input features.
    """
    try:
        df = pd.read_csv("discrepancies.csv")
    except FileNotFoundError:
        print("discrepancies.csv not found. Please run analyze_discrepancies.py first.")
        return

    # Calculate new features and handle potential division by zero
    df["case_1_miles_per_day"] = df["case_1_miles_traveled"] / df[
        "case_1_trip_duration_days"
    ].replace(0, 1)
    df["case_2_miles_per_day"] = df["case_2_miles_traveled"] / df[
        "case_2_trip_duration_days"
    ].replace(0, 1)
    df["case_1_receipts_per_day"] = df["case_1_total_receipts_amount"] / df[
        "case_1_trip_duration_days"
    ].replace(0, 1)
    df["case_2_receipts_per_day"] = df["case_2_total_receipts_amount"] / df[
        "case_2_trip_duration_days"
    ].replace(0, 1)
    df["case_1_mileage_receipt_ratio"] = df["case_1_miles_traveled"] / df[
        "case_1_total_receipts_amount"
    ].replace(0, 1e-6)
    df["case_2_mileage_receipt_ratio"] = df["case_2_miles_traveled"] / df[
        "case_2_total_receipts_amount"
    ].replace(0, 1e-6)

    # Set up the plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("Distribution of Features in Discrepant Cases", fontsize=20, y=1.02)
    axes = axes.flatten()

    features = {
        "trip_duration_days": "Trip Duration (Days)",
        "miles_traveled": "Miles Traveled",
        "total_receipts_amount": "Total Receipts Amount ($)",
        "miles_per_day": "Miles Per Day",
        "receipts_per_day": "Receipts Per Day ($)",
        "mileage_receipt_ratio": "Mileage-Receipt Ratio",
    }

    for i, (feature_col, title) in enumerate(features.items()):
        # Combine data from both cases into a single series for plotting
        combined_data = pd.concat(
            [df[f"case_1_{feature_col}"], df[f"case_2_{feature_col}"]],
            ignore_index=True,
        )

        sns.histplot(combined_data, ax=axes[i], kde=True, bins=30, color="purple")
        axes[i].set_title(title)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()

    output_filename = "discrepancy_histograms.png"
    plt.savefig(output_filename)
    print(f"Discrepancy histograms saved to {output_filename}")


if __name__ == "__main__":
    plot_discrepancies()
