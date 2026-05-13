from pathlib import Path
from collections import Counter

import pandas as pd

from freddie_schema import PERFORMANCE_COLUMNS


DATA_DIR = Path("data/raw/freddie_2023/extracted")
CHUNK_SIZE = 500_000


def main():
    performance_files = sorted(DATA_DIR.glob("historical_data_time_2023Q*.txt"))

    delinquency_counter = Counter()
    zero_balance_counter = Counter()

    min_reporting_period = None
    max_reporting_period = None
    total_rows = 0

    use_columns = [
        "monthly_reporting_period",
        "current_loan_delinquency_status",
        "zero_balance_code",
    ]

    for file_path in performance_files:
        print(f"\nProcessing: {file_path.name}")

        reader = pd.read_csv(
            file_path,
            sep="|",
            header=None,
            names=PERFORMANCE_COLUMNS,
            usecols=use_columns,
            dtype=str,
            chunksize=CHUNK_SIZE,
        )

        for chunk in reader:
            total_rows += len(chunk)

            delinquency_counter.update(
                chunk["current_loan_delinquency_status"]
                .fillna("MISSING")
                .astype(str)
                .value_counts()
                .to_dict()
            )

            zero_balance_counter.update(
                chunk["zero_balance_code"]
                .fillna("MISSING")
                .astype(str)
                .value_counts()
                .to_dict()
            )

            chunk_min = chunk["monthly_reporting_period"].min()
            chunk_max = chunk["monthly_reporting_period"].max()

            if min_reporting_period is None or chunk_min < min_reporting_period:
                min_reporting_period = chunk_min

            if max_reporting_period is None or chunk_max > max_reporting_period:
                max_reporting_period = chunk_max

    print("\n" + "=" * 80)
    print(f"Total performance rows processed: {total_rows:,}")
    print(f"Min monthly reporting period: {min_reporting_period}")
    print(f"Max monthly reporting period: {max_reporting_period}")

    print("\nDelinquency status counts:")
    for status, count in delinquency_counter.most_common():
        print(f"{status}: {count:,}")

    print("\nZero balance code counts:")
    for code, count in zero_balance_counter.most_common():
        print(f"{code}: {count:,}")


if __name__ == "__main__":
    main()