import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distributions():
    """
    Analyzes and plots the distributions of features for all cases vs. discrepant cases.
    """
    try:
        # Load datasets
        df_all = pd.read_csv("public_cases.csv")
        df_disc = pd.read_csv("discrepancies.csv")
    except FileNotFoundError as e:
        print(
            f"Error: {e.filename} not found. Please run previous analysis scripts first."
        )
        return

    # --- Prepare Full Dataset ---
    df_all["miles_per_day"] = df_all["miles_traveled"] / df_all[
        "trip_duration_days"
    ].replace(0, 1)
    df_all["receipts_per_day"] = df_all["total_receipts_amount"] / df_all[
        "trip_duration_days"
    ].replace(0, 1)
    df_all["mileage_receipt_ratio"] = df_all["miles_traveled"] / df_all[
        "total_receipts_amount"
    ].replace(0, 1)

    # --- Prepare Discrepant Dataset ---
    # Combine case 1 and case 2 into a single set of data points
    features_base = ["trip_duration_days", "miles_traveled", "total_receipts_amount"]
    features_derived = ["miles_per_day", "receipts_per_day", "mileage_receipt_ratio"]

    disc_data = {}
    for feature in features_base:
        disc_data[feature] = pd.concat(
            [df_disc[f"case_1_{feature}"], df_disc[f"case_2_{feature}"]],
            ignore_index=True,
        )

    disc_df = pd.DataFrame(disc_data)
    disc_df["miles_per_day"] = disc_df["miles_traveled"] / disc_df[
        "trip_duration_days"
    ].replace(0, 1)
    disc_df["receipts_per_day"] = disc_df["total_receipts_amount"] / disc_df[
        "trip_duration_days"
    ].replace(0, 1)
    disc_df["mileage_receipt_ratio"] = disc_df["miles_traveled"] / disc_df[
        "total_receipts_amount"
    ].replace(0, 1)

    all_features = {
        "trip_duration_days": "Trip Duration (Days)",
        "miles_traveled": "Miles Traveled",
        "total_receipts_amount": "Total Receipts Amount ($)",
        "miles_per_day": "Miles per Day",
        "receipts_per_day": "Receipts per Day ($)",
        "mileage_receipt_ratio": "Mileage-Receipt Ratio",
    }

    # --- Plot 1: Overlaid Histograms ---
    sns.set_style("whitegrid")
    fig1, axes1 = plt.subplots(3, 2, figsize=(15, 15))
    fig1.suptitle(
        "Distribution Comparison: All Cases vs. Discrepant Cases", fontsize=20, y=1.03
    )
    axes1 = axes1.flatten()

    for i, (feature_col, title) in enumerate(all_features.items()):
        # Set a common range for bins to make plots comparable
        combined_min = min(df_all[feature_col].min(), disc_df[feature_col].min())
        combined_max = max(df_all[feature_col].max(), disc_df[feature_col].max())
        # Use a sensible upper limit for visualization to handle extreme outliers
        quantile_max = df_all[feature_col].quantile(0.99)
        plot_max = min(combined_max, quantile_max * 1.5)

        bins = np.linspace(combined_min, plot_max, 50)

        sns.histplot(
            df_all[feature_col],
            ax=axes1[i],
            bins=bins,
            kde=False,
            stat="density",
            color="blue",
            alpha=0.6,
            label="All Cases",
        )
        sns.histplot(
            disc_df[feature_col],
            ax=axes1[i],
            bins=bins,
            kde=False,
            stat="density",
            color="red",
            alpha=0.6,
            label="Discrepant Cases",
        )
        axes1[i].set_title(title)
        axes1[i].set_xlabel("")
        axes1[i].set_ylabel("Density")
        axes1[i].set_xlim(combined_min, plot_max)
        axes1[i].legend()

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig1.savefig("feature_distribution_comparison.png", dpi=300)
    plt.close(fig1)

    # --- Plot 2: Ratio of Distributions ---
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 15))
    fig2.suptitle(
        "Ratio of Distributions (Discrepant vs. All Cases)", fontsize=20, y=1.03
    )
    axes2 = axes2.flatten()

    for i, (feature_col, title) in enumerate(all_features.items()):
        combined_min = min(df_all[feature_col].min(), disc_df[feature_col].min())
        quantile_max = df_all[feature_col].quantile(0.99)
        plot_max = min(df_all[feature_col].max(), quantile_max * 1.5)

        bins = np.linspace(combined_min, plot_max, 30)

        # Calculate histograms (proportions)
        hist_all, _ = np.histogram(df_all[feature_col], bins=bins, density=False)
        hist_disc, _ = np.histogram(disc_df[feature_col], bins=bins, density=False)

        prop_all = hist_all / len(df_all)
        prop_disc = hist_disc / len(disc_df)

        # Calculate ratio, handle division by zero
        ratio = prop_disc / (prop_all + 1e-10)

        bin_centers = (bins[:-1] + bins[1:]) / 2

        axes2[i].bar(bin_centers, ratio, width=(bins[1] - bins[0]) * 0.9, color="green")
        axes2[i].axhline(1, color="grey", linestyle="--")
        axes2[i].set_title(title)
        axes2[i].set_xlabel(title)
        axes2[i].set_ylabel("Representation Ratio")
        axes2[i].set_xlim(combined_min, plot_max)

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig2.savefig("feature_distribution_ratios.png", dpi=300)
    plt.close(fig2)

    print(
        "Plots saved to feature_distribution_comparison.png and feature_distribution_ratios.png"
    )


if __name__ == "__main__":
    plot_feature_distributions()
