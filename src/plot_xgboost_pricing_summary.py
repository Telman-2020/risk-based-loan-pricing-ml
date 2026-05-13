from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path("reports/xgboost_pricing_summary_by_risk_band.csv")
OUTPUT_DIR = Path("reports/figures/model")
OUTPUT_PATH = OUTPUT_DIR / "xgboost_pricing_summary_by_risk_band.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_PATH)

    band_order = ["Low risk", "Medium risk", "High risk"]
    df["risk_band"] = pd.Categorical(
        df["risk_band"],
        categories=band_order,
        ordered=True,
    )
    df = df.sort_values("risk_band")

    plt.figure(figsize=(9, 6))

    plt.plot(
        df["risk_band"],
        df["average_pd"],
        marker="o",
        label="Average calibrated PD",
    )

    plt.plot(
        df["risk_band"],
        df["actual_bad_rate"],
        marker="o",
        label="Actual bad-loan rate",
    )

    plt.plot(
        df["risk_band"],
        df["average_expected_loss"],
        marker="o",
        label="Average expected loss",
    )

    plt.plot(
        df["risk_band"],
        df["average_suggested_apr"],
        marker="o",
        label="Average suggested APR",
    )

    plt.title("XGBoost Risk-Based Pricing Summary by Risk Band")
    plt.xlabel("Risk band")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print("Saved XGBoost pricing summary plot:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()