from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path("reports/logistic_regression_test_predictions.csv")
SELECTED_THRESHOLD_PATH = Path("reports/selected_threshold.csv")
OUTPUT_PATH = Path("reports/pricing_decisions.csv")
SUMMARY_OUTPUT_PATH = Path("reports/pricing_summary_by_risk_band.csv")


LGD = 0.40
FUNDING_COST = 0.04
OPERATING_COST = 0.02
PROFIT_MARGIN = 0.03


def assign_risk_band(calibrated_pd: float) -> str:
    if calibrated_pd < 0.015:
        return "Low risk"
    if calibrated_pd < 0.03:
        return "Medium risk"
    return "High risk"


def assign_decision(calibrated_pd: float, threshold: float) -> str:
    if calibrated_pd >= threshold:
        return "Refer or decline"
    return "Approve"


def calculate_expected_loss(calibrated_pd: float) -> float:
    return calibrated_pd * LGD


def calculate_suggested_apr(calibrated_pd: float) -> float:
    expected_loss = calculate_expected_loss(calibrated_pd)
    return FUNDING_COST + OPERATING_COST + PROFIT_MARGIN + expected_loss


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    selected_threshold_df = pd.read_csv(SELECTED_THRESHOLD_PATH)

    selected_threshold = float(selected_threshold_df["threshold"].iloc[0])

    df["risk_band"] = df["calibrated_pd"].apply(assign_risk_band)
    df["decision"] = df["calibrated_pd"].apply(
        lambda pd_value: assign_decision(pd_value, selected_threshold)
    )

    df["expected_loss"] = df["calibrated_pd"].apply(calculate_expected_loss)
    df["suggested_apr"] = df["calibrated_pd"].apply(calculate_suggested_apr)

    output_columns = [
        "calibrated_pd",
        "risk_band",
        "decision",
        "expected_loss",
        "suggested_apr",
        "actual_bad_loan",
    ]

    pricing_df = df[output_columns].copy()

    pricing_df[
        [
            "calibrated_pd",
            "expected_loss",
            "suggested_apr",
        ]
    ] = pricing_df[
        [
            "calibrated_pd",
            "expected_loss",
            "suggested_apr",
        ]
    ].round(4)

    pricing_df.to_csv(OUTPUT_PATH, index=False)

    print("Pricing decisions preview:")
    print(pricing_df.head(20))

    print("\nPricing summary by risk band:")
    summary = (
        pricing_df.groupby("risk_band")
        .agg(
            loans=("actual_bad_loan", "count"),
            bad_loans=("actual_bad_loan", "sum"),
            average_pd=("calibrated_pd", "mean"),
            average_expected_loss=("expected_loss", "mean"),
            average_suggested_apr=("suggested_apr", "mean"),
        )
        .reset_index()
    )

    summary[
        [
            "average_pd",
            "average_expected_loss",
            "average_suggested_apr",
        ]
    ] = summary[
        [
            "average_pd",
            "average_expected_loss",
            "average_suggested_apr",
        ]
    ].round(4)

    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print(summary)

    print("\nSaved pricing summary:")
    print(SUMMARY_OUTPUT_PATH)

    print("\nSaved pricing decisions:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()