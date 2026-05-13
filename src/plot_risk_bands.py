from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RISK_BAND_PATH = Path("reports/risk_band_analysis.csv")
OUTPUT_DIR = Path("reports/figures/model")
OUTPUT_PATH = OUTPUT_DIR / "risk_band_analysis.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RISK_BAND_PATH)

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
        df["actual_bad_rate"],
        marker="o",
        label="Actual bad-loan rate",
    )

    plt.plot(
        df["risk_band"],
        df["average_pd"],
        marker="o",
        label="Average calibrated PD",
    )

    plt.title("Risk Band Calibration Analysis")
    plt.xlabel("Risk band")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print("Saved risk band plot:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()