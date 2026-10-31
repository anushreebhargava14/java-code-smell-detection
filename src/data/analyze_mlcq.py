"""
Stage 1, Step 3: Understand smell/severity distribution and the real scale
of source-code retrieval Stage 2 will require.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SAMPLES_FILE = RAW_DATA_DIR / "MLCQCodeSmellSamples.csv"


def main():
    df = pd.read_csv(SAMPLES_FILE, sep=";")

    print("=== Smell type counts ===")
    print(df["smell"].value_counts())
    print()

    print("=== Severity counts (overall) ===")
    print(df["severity"].value_counts())
    print()

    print("=== Severity counts per smell ===")
    print(df.groupby("smell")["severity"].value_counts())
    print()

    print("=== type (class vs function) counts ===")
    print(df["type"].value_counts())
    print()

    print("=== industry_relevant_project flag ===")
    print(df["is_from_industry_relevant_project"].value_counts())
    print()

    print(f"Unique repositories: {df['repository'].nunique()}")
    print(f"Unique (repository, commit_hash) pairs: "
          f"{df[['repository', 'commit_hash']].drop_duplicates().shape[0]}")
    print(f"Unique (repository, commit_hash, path) files referenced: "
          f"{df[['repository', 'commit_hash', 'path']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()