from pathlib import Path


DATA_DIR = Path("data/raw/freddie_2023/extracted")


def count_rows(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        return sum(1 for _ in file)


def main():
    files = sorted(DATA_DIR.glob("*.txt"))

    total_origination_rows = 0
    total_performance_rows = 0

    for file_path in files:
        row_count = count_rows(file_path)

        if "_time_" in file_path.name:
            file_type = "performance"
            total_performance_rows += row_count
        else:
            file_type = "origination"
            total_origination_rows += row_count

        print(f"{file_path.name}: {row_count:,} rows ({file_type})")

    print("\nTotals:")
    print(f"Origination rows: {total_origination_rows:,}")
    print(f"Performance rows: {total_performance_rows:,}")


if __name__ == "__main__":
    main()