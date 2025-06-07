import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def analyze_results(filepath="evaluation_results_with_features.csv"):
    """
    Reads the evaluation results and creates plots to analyze model errors.

    Args:
        filepath (str): The path to the evaluation results CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please run 'python solution.py' first to generate the results.")
        sys.exit(1)

    print(f"📄 Loaded {len(df)} records from '{filepath}'")

    # Set the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 5))

    # Plot 1: Receipts vs. Error
    plt.subplot(1, 3, 1)
    sns.scatterplot(
        x="total_receipts_amount", y="error", data=df, alpha=0.5, color="blue"
    )
    plt.title("Receipt Amount vs. Prediction Error")
    plt.xlabel("Total Receipts Amount ($)")
    plt.ylabel("Absolute Error ($)")

    # Plot 2: Mileage vs. Error
    plt.subplot(1, 3, 2)
    sns.scatterplot(x="miles_traveled", y="error", data=df, alpha=0.5, color="green")
    plt.title("Miles Traveled vs. Prediction Error")
    plt.xlabel("Miles Traveled")
    plt.ylabel("Absolute Error ($)")

    # Plot 3: Duration vs. Error
    plt.subplot(1, 3, 3)
    sns.scatterplot(x="trip_duration_days", y="error", data=df, alpha=0.5, color="red")
    plt.title("Trip Duration vs. Prediction Error")
    plt.xlabel("Trip Duration (days)")
    plt.ylabel("Absolute Error ($)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_results()
