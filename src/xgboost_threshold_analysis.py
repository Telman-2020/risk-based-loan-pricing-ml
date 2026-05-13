from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


PREDICTIONS_PATH = Path("reports/xgboost_test_predictions.csv")

THRESHOLD_OUTPUT_PATH = Path("reports/xgboost_threshold_analysis.csv")
SELECTED_THRESHOLD_OUTPUT_PATH = Path("reports/xgboost_selected_threshold.csv")

FIGURES_DIR = Path("reports/figures/model")
THRESHOLD_PLOT_PATH = FIGURES_DIR / "xgboost_threshold_analysis.png"

MIN_APPROVAL_RATE = 0.80


def calculate_threshold_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    true_negative = ((y_true == 0) & (y_pred == 0)).sum()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    approval_rate = (y_pred == 0).mean()
    decline_rate = (y_pred == 1).mean()

    false_positive_rate = false_positive / (false_positive + true_negative)

    return {
        "threshold": round(threshold, 3),
        "approval_rate": round(approval_rate, 4),
        "decline_rate": round(decline_rate, 4),
        "precision": round(precision, 4),
        "recall_bad_loan_capture_rate": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def select_best_threshold(results_df):
    eligible_df = results_df[
        results_df["approval_rate"] >= MIN_APPROVAL_RATE
    ].copy()

    if eligible_df.empty:
        selected = results_df.sort_values(
            ["f1_score", "recall_bad_loan_capture_rate"],
            ascending=False,
        ).head(1)

        selection_rule = "maximise F1 because no threshold met approval-rate rule"

    else:
        selected = eligible_df.sort_values(
            [
                "recall_bad_loan_capture_rate",
                "precision",
                "f1_score",
            ],
            ascending=False,
        ).head(1)

        selection_rule = (
            f"maximise bad-loan recall while approval_rate >= {MIN_APPROVAL_RATE}"
        )

    selected = selected.copy()
    selected["selection_rule"] = selection_rule

    return selected


def plot_threshold_analysis(results_df, selected_threshold):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["threshold"],
        results_df["approval_rate"],
        marker="o",
        label="Approval rate",
    )

    plt.plot(
        results_df["threshold"],
        results_df["decline_rate"],
        marker="o",
        label="Decline rate",
    )

    plt.plot(
        results_df["threshold"],
        results_df["recall_bad_loan_capture_rate"],
        marker="o",
        label="Bad-loan capture rate",
    )

    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        marker="o",
        label="Precision",
    )

    plt.axvline(
        x=selected_threshold,
        linestyle="--",
        label=f"Selected threshold = {selected_threshold:.3f}",
    )

    plt.title("XGBoost Threshold Analysis Using Calibrated PD")
    plt.xlabel("Calibrated PD threshold")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(THRESHOLD_PLOT_PATH, dpi=150)
    plt.close()

    print("\nSaved threshold plot:")
    print(THRESHOLD_PLOT_PATH)


def main():
    print("Loading XGBoost predictions...")
    df = pd.read_csv(PREDICTIONS_PATH)

    y_true = df["actual_bad_loan"].astype(int)
    y_score = df["calibrated_pd"]

    thresholds = np.arange(0.001, 0.201, 0.001)

    rows = [
        calculate_threshold_metrics(y_true, y_score, threshold)
        for threshold in thresholds
    ]

    results_df = pd.DataFrame(rows)
    selected_threshold_df = select_best_threshold(results_df)

    selected_threshold = float(selected_threshold_df["threshold"].iloc[0])

    results_df.to_csv(THRESHOLD_OUTPUT_PATH, index=False)
    selected_threshold_df.to_csv(SELECTED_THRESHOLD_OUTPUT_PATH, index=False)

    plot_threshold_analysis(
        results_df=results_df,
        selected_threshold=selected_threshold,
    )

    print("\nAutomatically selected XGBoost threshold:")
    print(selected_threshold_df)

    print("\nSaved threshold analysis:")
    print(THRESHOLD_OUTPUT_PATH)

    print("\nSaved selected threshold:")
    print(SELECTED_THRESHOLD_OUTPUT_PATH)


if __name__ == "__main__":
    main()