"""
Stage 3.3, Step 1: Select ~25 representative snippets spanning both
type levels, all 4 smells, and a range of sizes, for tool comparison testing.
"""

import pandas as pd
from pathlib import Path

INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
JOINED_OUTPUT = INTERIM_DIR / "labeled_rows_with_files.csv"
TEST_SAMPLE_OUTPUT = INTERIM_DIR / "tool_test_sample.csv"


def main():
    df = pd.read_csv(JOINED_OUTPUT)
    valid = df[df["snippet_path"].notna()].copy()
    valid["snippet_length"] = valid["end_line"] - valid["start_line"] + 1

    samples = []
    for smell, group in valid.groupby("smell"):
        # For each smell, grab shortest, a middle-length, longest, and one extra
        sorted_group = group.sort_values("snippet_length")
        picks = pd.concat([
            sorted_group.head(2),                                    # shortest
            sorted_group.iloc[[len(sorted_group) // 2]],              # middle
            sorted_group.tail(2),                                     # longest
        ]).drop_duplicates(subset="id")
        samples.append(picks)

    test_sample = pd.concat(samples).drop_duplicates(subset="id")
    test_sample.to_csv(TEST_SAMPLE_OUTPUT, index=False)

    print(f"Selected {len(test_sample)} snippets for tool testing.")
    print(test_sample[["id", "smell", "type", "snippet_length", "snippet_path"]].to_string())


if __name__ == "__main__":
    main()