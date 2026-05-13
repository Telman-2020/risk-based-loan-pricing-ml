from pathlib import Path

import pandas as pd

from freddie_schema import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS


DATA_DIR = Path("data/raw/freddie_2023/extracted")


def get_columns(file_path: Path):
    if "_time_" in file_path.name:
        return PERFORMANCE_COLUMNS

    return ORIGINATION_COLUMNS


def inspect_file(file_path: Path, nrows: int = 5):
    columns = get_columns(file_path)

    print("=" * 100)
    print(f"File: {file_path.name}")
    print(f"Size MB: {file_path.stat().st_size / 1024 / 1024:.2f}")

    df = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=columns,
        nrows=nrows,
        dtype=str,
    )

    print(f"Preview shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())


def main():
    files = sorted(DATA_DIR.glob("*.txt"))

    print(f"Found {len(files)} text files\n")

    for file_path in files:
        inspect_file(file_path)


if __name__ == "__main__":
    main()