"""
Stage 1, Step 2: Inspect the real structure of MLCQCodeSmellSamples.csv.
No assumptions about columns — we print exactly what's there.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SAMPLES_FILE = RAW_DATA_DIR / "MLCQCodeSmellSamples.csv"


def main():
    # MLCQ's CSV has been reported using ';' as separator in some
    # reproductions — we try comma first, and fall back if only
    # one column is detected (a sign the separator was wrong).
    try:
        df = pd.read_csv(SAMPLES_FILE, sep=";")
        if df.shape[1] == 1:
            raise ValueError("Only 1 column with ';' — wrong separator")
    except (pd.errors.ParserError, ValueError):
        print("';' separator didn't work cleanly — retrying with ','")
        df = pd.read_csv(SAMPLES_FILE, sep=",")

    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("Column names and dtypes:")
    print(df.dtypes)
    print()

    print("First 3 rows:")
    print(df.head(3).to_string())
    print()

    print("Missing values per column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()