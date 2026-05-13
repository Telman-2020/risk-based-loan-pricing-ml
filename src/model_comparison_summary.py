from pathlib import Path

import pandas as pd


LOGISTIC_RESULTS = {
    "model_name": "Logistic Regression L1",
    "model_type": "Interpretable baseline",
    "roc_auc": 0.8165,
    "pr_auc": 0.1130,
    "notes": "Scorecard-style model; calibrated PD used for pricing",
}

MLP_RESULTS = {
    "model_name": "Simple MLP",
    "model_type": "Simple deep learning challenger",
    "roc_auc": 0.8213,
    "pr_auc": 0.1161,
    "notes": "Improved slightly over Logistic Regression but weaker than XGBoost",
}

XGBOOST_RESULTS = {
    "model_name": "XGBoost",
    "model_type": "Best challenger model",
    "roc_auc": 0.8248,
    "pr_auc": 0.1313,
    "notes": "Best PR-AUC; selected as strongest model for risk-based pricing",
}


OUTPUT_PATH = Path("reports/final_model_comparison.csv")


def main():
    results_df = pd.DataFrame(
        [
            LOGISTIC_RESULTS,
            MLP_RESULTS,
            XGBOOST_RESULTS,
        ]
    )

    results_df = results_df.sort_values("pr_auc", ascending=False)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("Final model comparison:")
    print(results_df)

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()