from pathlib import Path

import pandas as pd


FEATURES_PATH = Path("data/processed/freddie_2023_features.csv")
TARGET_PATH = Path("data/processed/freddie_2023_target.csv")
OUTPUT_PATH = Path("data/processed/freddie_2023_model_dataset.csv")


def main():
    print("Loading features...")
    features_df = pd.read_csv(FEATURES_PATH, dtype=str)

    print("Loading target...")
    target_df = pd.read_csv(TARGET_PATH, dtype=str)

    print("\nFeatures shape:")
    print(features_df.shape)

    print("\nTarget shape:")
    print(target_df.shape)

    model_df = features_df.merge(
        target_df,
        on="loan_sequence_number",
        how="inner",
    )

    print("\nModel dataset shape:")
    print(model_df.shape)

    print("\nBad loan distribution:")
    print(model_df["bad_loan"].value_counts())

    print("\nBad loan distribution percentage:")
    print(model_df["bad_loan"].value_counts(normalize=True).round(4))

    print("\nDuplicate loan IDs:")
    print(model_df["loan_sequence_number"].duplicated().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved model dataset:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()