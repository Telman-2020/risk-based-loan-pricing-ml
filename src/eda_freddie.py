from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_DATA_PATH = Path("data/processed/freddie_2023_model_dataset.csv")
FIGURES_DIR = Path("reports/figures/eda")
CROSS_PLOTS_DIR = FIGURES_DIR / "cross_plots"
REPORTS_DIR = Path("reports")


NUMERIC_FEATURES = [
    "credit_score",
    "mortgage_insurance_percentage",
    "number_of_units",
    "original_cltv",
    "original_dti_ratio",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
    "max_delinquency_status",
    "observation_months",
]

CATEGORICAL_FEATURES = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "property_state",
    "property_type",
    "loan_purpose",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
]

TARGET = "bad_loan"


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).lower()


def get_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df[NUMERIC_FEATURES].copy()

    for col in NUMERIC_FEATURES:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    return numeric_df


def save_numeric_histograms(df: pd.DataFrame) -> None:
    print("\nCreating numerical histograms...")

    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        plt.figure(figsize=(8, 5))
        plt.hist(series, bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()

        output_path = FIGURES_DIR / f"hist_{safe_filename(col)}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")


def save_categorical_bar_plots(df: pd.DataFrame, top_n: int = 20) -> None:
    print("\nCreating categorical bar plots...")

    for col in CATEGORICAL_FEATURES:
        counts = (
            df[col]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(top_n)
            .sort_values()
        )

        plt.figure(figsize=(9, 6))
        plt.barh(counts.index, counts.values)
        plt.title(f"Top {top_n} categories for {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()

        output_path = FIGURES_DIR / f"bar_{safe_filename(col)}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")


def save_correlation_heatmap(numeric_df: pd.DataFrame) -> pd.DataFrame:
    print("\nCreating Pearson correlation heatmap...")

    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    image = plt.imshow(corr, aspect="auto")
    plt.colorbar(image)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)

    plt.title("Pearson Correlation Heatmap - Numerical Features")
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "correlation_heatmap_numeric_features.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")

    return corr


def get_correlation_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    pairs = []
    columns = corr.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            correlation = corr.loc[feature_1, feature_2]

            if pd.notna(correlation) and abs(correlation) > threshold:
                pairs.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "pearson_correlation": round(correlation, 2),
                        "absolute_correlation": round(abs(correlation), 2),
                    }
                )

    pairs_df = pd.DataFrame(pairs)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            "absolute_correlation",
            ascending=False,
        )

    return pairs_df


def save_high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    print(f"\nFinding numerical feature pairs with |Pearson correlation| > {threshold}...")

    pairs_df = get_correlation_pairs(corr, threshold=threshold)

    output_path = REPORTS_DIR / "high_correlation_numeric_pairs.csv"
    pairs_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    print("\nHigh-correlation numerical pairs:")
    if pairs_df.empty:
        print("No pairs found above threshold.")
    else:
        print(pairs_df)

    return pairs_df


def save_cross_plots_for_high_correlations(
    numeric_df: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    sample_size: int = 50_000,
) -> None:
    print("\nCreating cross plots for high-correlation numerical pairs...")

    if high_corr_pairs.empty:
        print("No high-correlation pairs found, so no cross plots were created.")
        return

    plot_df = numeric_df.copy()

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    for _, row in high_corr_pairs.iterrows():
        feature_1 = row["feature_1"]
        feature_2 = row["feature_2"]
        corr_value = row["pearson_correlation"]

        pair_df = plot_df[[feature_1, feature_2]].dropna()

        plt.figure(figsize=(8, 6))
        plt.scatter(pair_df[feature_1], pair_df[feature_2], alpha=0.25, s=8)
        plt.title(f"{feature_1} vs {feature_2} | Pearson r = {corr_value}")
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.grid(True)
        plt.tight_layout()

        output_path = (
            CROSS_PLOTS_DIR
            / f"cross_{safe_filename(feature_1)}_vs_{safe_filename(feature_2)}.png"
        )

        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")


def save_spearman_nonlinear_pairs(numeric_df: pd.DataFrame) -> None:
    print("\nCreating Spearman non-linear/monotonic correlation pairs CSV...")

    spearman_corr = numeric_df.corr(method="spearman")

    pairs = []
    columns = spearman_corr.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            correlation = spearman_corr.loc[feature_1, feature_2]

            pairs.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "spearman_correlation": round(correlation, 2),
                    "absolute_spearman_correlation": round(abs(correlation), 2),
                }
            )

    pairs_df = pd.DataFrame(pairs).sort_values(
        "absolute_spearman_correlation",
        ascending=False,
    )

    output_path = REPORTS_DIR / "nonlinear_spearman_numeric_pairs_all.csv"
    pairs_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("\nTop 20 Spearman non-linear/monotonic pairs:")
    print(pairs_df.head(20))


def save_target_distribution(df: pd.DataFrame) -> None:
    print("\nCreating target distribution plot...")

    counts = df[TARGET].astype(int).value_counts().sort_index()
    labels = ["Good loan", "Bad loan"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts.values)
    plt.title("Target Distribution")
    plt.xlabel("Target class")
    plt.ylabel("Number of loans")
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "target_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")

    print("\nTarget distribution:")
    print(counts)
    print("\nTarget distribution percentage:")
    print((counts / counts.sum()).round(4))


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model dataset...")
    df = pd.read_csv(MODEL_DATA_PATH)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    numeric_df = get_numeric_df(df)

    save_target_distribution(df)
    save_numeric_histograms(df)
    save_categorical_bar_plots(df)

    pearson_corr = save_correlation_heatmap(numeric_df)

    high_corr_pairs = save_high_correlation_pairs(
        pearson_corr,
        threshold=0.50,
    )

    save_cross_plots_for_high_correlations(
        numeric_df,
        high_corr_pairs,
        sample_size=50_000,
    )

    save_spearman_nonlinear_pairs(numeric_df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()