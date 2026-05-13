from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path("reports/xgboost_test_predictions.csv")
OUTPUT_PATH = Path("reports/xgboost_risk_band_analysis.csv")


def assign_risk_band(calibrated_pd: float) -> str:
    if calibrated_pd < 0.012:
        return "Low risk"
    if calibrated_pd < 0.024:
        return "Medium risk"
    return "High risk"


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    df["risk_band"] = df["calibrated_pd"].apply(assign_risk_band)

    summary = (
        df.groupby("risk_band")
        .agg(
            loans=("actual_bad_loan", "count"),
            bad_loans=("actual_bad_loan", "sum"),
            average_pd=("calibrated_pd", "mean"),
            min_pd=("calibrated_pd", "min"),
            max_pd=("calibrated_pd", "max"),
        )
        .reset_index()
    )

    summary["loan_share"] = summary["loans"] / summary["loans"].sum()
    summary["actual_bad_rate"] = summary["bad_loans"] / summary["loans"]

    summary = summary[
        [
            "risk_band",
            "loans",
            "loan_share",
            "bad_loans",
            "actual_bad_rate",
            "average_pd",
            "min_pd",
            "max_pd",
        ]
    ]

    summary = summary.sort_values(
        "risk_band",
        key=lambda col: col.map(
            {
                "Low risk": 0,
                "Medium risk": 1,
                "High risk": 2,
            }
        ),
    )

    numeric_cols = [
        "loan_share",
        "actual_bad_rate",
        "average_pd",
        "min_pd",
        "max_pd",
    ]

    summary[numeric_cols] = summary[numeric_cols].round(4)

    summary.to_csv(OUTPUT_PATH, index=False)

    print("XGBoost risk band analysis:")
    print(summary)

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()