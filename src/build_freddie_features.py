from pathlib import Path

import pandas as pd

from freddie_schema import ORIGINATION_COLUMNS


DATA_DIR = Path("data/raw/freddie_2023/extracted")
OUTPUT_PATH = Path("data/processed/freddie_2023_features.csv")


def main():
    origination_files = sorted(DATA_DIR.glob("historical_data_2023Q*.txt"))

    dataframes = []

    for file_path in origination_files:
        print(f"Reading: {file_path.name}")

        df = pd.read_csv(
            file_path,
            sep="|",
            header=None,
            names=ORIGINATION_COLUMNS,
            dtype=str,
        )

        df["source_file"] = file_path.name
        dataframes.append(df)

    features_df = pd.concat(dataframes, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved feature table:")
    print(OUTPUT_PATH)

    print("\nFeature table shape:")
    print(features_df.shape)

    print("\nFirst 5 rows:")
    print(features_df.head())

    print("\nMissing values - top 20:")
    print(features_df.isna().sum().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()