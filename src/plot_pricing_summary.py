from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PRICING_DECISIONS_PATH = Path("reports/pricing_decisions.csv")
OUTPUT_DIR = Path("reports/figures/model")
OUTPUT_PATH = OUTPUT_DIR / "pricing_summary_by_risk_band.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PRICING_DECISIONS_PATH)

    summary = (
        df.groupby("risk_band")
        .agg(
            loans=("actual_bad_loan", "count"),
            bad_loans=("actual_bad_loan", "sum"),
            average_pd=("calibrated_pd", "mean"),
            average_expected_loss=("expected_loss", "mean"),
            average_suggested_apr=("suggested_apr", "mean"),
        )
        .reset_index()
    )

    band_order = ["Low risk", "Medium risk", "High risk"]
    summary["risk_band"] = pd.Categorical(
        summary["risk_band"],
        categories=band_order,
        ordered=True,
    )
    summary = summary.sort_values("risk_band")

    print("Pricing summary used for plot:")
    print(summary)

    plt.figure(figsize=(9, 6))

    plt.plot(
        summary["risk_band"],
        summary["average_pd"],
        marker="o",
        label="Average calibrated PD",
    )

    plt.plot(
        summary["risk_band"],
        summary["average_expected_loss"],
        marker="o",
        label="Average expected loss",
    )

    plt.plot(
        summary["risk_band"],
        summary["average_suggested_apr"],
        marker="o",
        label="Average suggested APR",
    )

    plt.title("Risk-Based Pricing Summary by Risk Band")
    plt.xlabel("Risk band")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print("\nSaved pricing summary plot:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()