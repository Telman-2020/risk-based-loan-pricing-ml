from pathlib import Path

import pandas as pd

from freddie_schema import PERFORMANCE_COLUMNS


DATA_DIR = Path("data/raw/freddie_2023/extracted")
OUTPUT_PATH = Path("data/processed/freddie_2023_target.csv")
CHUNK_SIZE = 500_000


def clean_delinquency_status(value):
    """
    Convert Freddie Mac delinquency status to numeric months delinquent.

    Examples:
    0 = current
    1 = 30 days delinquent
    2 = 60 days delinquent
    3 = 90 days delinquent

    Non-numeric values like RA are treated separately as missing here.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main():
    performance_files = sorted(DATA_DIR.glob("historical_data_time_2023Q*.txt"))

    loan_targets = {}

    use_columns = [
        "loan_sequence_number",
        "monthly_reporting_period",
        "current_loan_delinquency_status",
        "zero_balance_code",
    ]

    for file_path in performance_files:
        print(f"Processing: {file_path.name}")

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
            chunk["delinquency_numeric"] = chunk[
                "current_loan_delinquency_status"
            ].apply(clean_delinquency_status)

            chunk["is_90_plus_delinquent"] = (
                chunk["delinquency_numeric"].fillna(0) >= 3
            ).astype(int)

            grouped = chunk.groupby("loan_sequence_number").agg(
                max_delinquency_status=("delinquency_numeric", "max"),
                ever_90_plus_delinquent=("is_90_plus_delinquent", "max"),
                first_reporting_period=("monthly_reporting_period", "min"),
                last_reporting_period=("monthly_reporting_period", "max"),
                observation_months=("monthly_reporting_period", "nunique"),
            )

            for loan_id, row in grouped.iterrows():
                if loan_id not in loan_targets:
                    loan_targets[loan_id] = row.to_dict()
                else:
                    loan_targets[loan_id]["max_delinquency_status"] = max(
                        loan_targets[loan_id]["max_delinquency_status"],
                        row["max_delinquency_status"],
                    )
                    loan_targets[loan_id]["ever_90_plus_delinquent"] = max(
                        loan_targets[loan_id]["ever_90_plus_delinquent"],
                        row["ever_90_plus_delinquent"],
                    )
                    loan_targets[loan_id]["first_reporting_period"] = min(
                        loan_targets[loan_id]["first_reporting_period"],
                        row["first_reporting_period"],
                    )
                    loan_targets[loan_id]["last_reporting_period"] = max(
                        loan_targets[loan_id]["last_reporting_period"],
                        row["last_reporting_period"],
                    )
                    loan_targets[loan_id]["observation_months"] += row[
                        "observation_months"
                    ]

    target_df = (
        pd.DataFrame.from_dict(loan_targets, orient="index")
        .reset_index()
        .rename(columns={"index": "loan_sequence_number"})
    )

    target_df = target_df.rename(
        columns={"ever_90_plus_delinquent": "bad_loan"}
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved target table:")
    print(OUTPUT_PATH)

    print("\nTarget table shape:")
    print(target_df.shape)

    print("\nBad loan distribution:")
    print(target_df["bad_loan"].value_counts())

    print("\nBad loan distribution percentage:")
    print(target_df["bad_loan"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()