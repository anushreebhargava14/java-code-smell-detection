"""
Stage 3.3 (revised): print the REAL schemas of CK output and our label tables,
so the full-file mapping is designed against actual columns.
Read-only: changes nothing.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
INTERIM = PROJECT_ROOT / "data" / "interim"
CK_WRAPPED = INTERIM / "tool_test_workdir" / "ck_output_wrapped"
CK_AS_IS = INTERIM / "tool_test_workdir" / "ck_output_as_is"


def show(title: str, path: Path, cols_of_interest=None, rows=3):
    print("=" * 80)
    print(title, "->", path)
    if not path.exists():
        print("  MISSING")
        return
    df = pd.read_csv(path)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    if cols_of_interest:
        present = [c for c in cols_of_interest if c in df.columns]
        if present and len(df):
            print(df[present].head(rows).to_string(index=False))
    print()


def main():
    show("MLCQ labeled rows", INTERIM / "labeled_rows_with_files.csv",
         ["id", "smell", "type", "code_name", "start_line", "end_line", "file_id", "label"])
    show("Fetch manifest", RAW / "fetch_manifest.csv",
         ["file_id", "status", "path"])

    for label, folder in (("wrapped", CK_WRAPPED), ("as_is", CK_AS_IS)):
        print(f"--- CK output folder ({label}): {folder}")
        if folder.exists():
            print("  files:", sorted(p.name for p in folder.iterdir()))
        show(f"CK class.csv ({label})", folder / "class.csv",
             ["file", "class", "type", "loc", "wmc", "cbo", "lcom"])
        show(f"CK method.csv ({label})", folder / "method.csv",
             ["file", "class", "method", "constructor", "line", "loc", "wmc"], rows=5)


if __name__ == "__main__":
    main()