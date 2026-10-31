"""
Stage 3.1 & 3.2 verification: print several real extracted snippets
alongside their metadata so we can visually confirm line ranges/scope are correct.
"""

import pandas as pd
from pathlib import Path

INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
JOINED_OUTPUT = INTERIM_DIR / "labeled_rows_with_files.csv"
PROJECT_ROOT = INTERIM_DIR.parents[0]


def main():
    df = pd.read_csv(JOINED_OUTPUT)
    valid = df[df["snippet_path"].notna()]

    # Sample a few rows across different smell types for inspection
    sample = valid.groupby("smell").head(2)

    for _, row in sample.iterrows():
        print("=" * 80)
        print(f"row id: {row['id']}  smell: {row['smell']}  severity: {row['severity']}  "
              f"label: {row['label']}  type: {row['type']}")
        print(f"code_name: {row['code_name']}")
        print(f"lines: {row['start_line']}-{row['end_line']}")
        snippet_path = PROJECT_ROOT / row["snippet_path"]
        content = snippet_path.read_text(encoding="utf-8")
        print(f"snippet line count: {len(content.splitlines())}")
        print("--- snippet content ---")
        print(content)
        print()


if __name__ == "__main__":
    main()