"""
One-off, read-only inspection of MLCQ row id 3006 and CK's method.csv
for that file, to understand the 'Class.method Param' naming pattern.
Does not modify the working mapping script.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM = PROJECT_ROOT / "data" / "interim"
SAMPLE_FILE = INTERIM / "tool_test_sample.csv"
CK_OUTPUT = INTERIM / "ck_fullfile_test_output"

row = pd.read_csv(SAMPLE_FILE)
row = row[row["id"] == 3006]
print(row[["id", "smell", "type", "code_name", "start_line", "end_line", "file_id"]].to_string(index=False))

file_id = int(row.iloc[0]["file_id"])
method_df = pd.read_csv(CK_OUTPUT / "method.csv")
matches = method_df[method_df["file"].str.contains(f"{file_id}.java", regex=False, na=False)]
matches = matches[matches["method"].str.split("/").str[0] == "isInfinite"]
print("\nAll CK 'isInfinite' methods in this file:")
print(matches[["class", "method", "line"]].to_string(index=False))