"""
Combine every successful CK batch/sub-batch output into one class.csv
and one method.csv. File 1760.java is excluded (isolated OOM cause).
Empty (0-byte) CSVs from failed CK runs are skipped.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = PROJECT_ROOT / "data" / "interim" / "ck_batches"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "ck_full_combined"
SOURCE_FILES_DIR = PROJECT_ROOT / "data" / "raw" / "source_files"
EXCLUDED_FILE_ID = 1760

DEDUP_KEYS = {
    "class.csv": ["file_id", "class"],
    "method.csv": ["file_id", "class", "method", "line"],
}


def read_ck_csv(path: Path):
    """Return a DataFrame, or None if the file is empty/unreadable."""
    try:
        if path.stat().st_size == 0:
            return None
        df = pd.read_csv(path)
        return None if df.empty else df
    except pd.errors.EmptyDataError:
        return None


def combine(csv_name: str):
    frames, skipped = [], []
    for csv_path in sorted(BATCH_ROOT.rglob(csv_name)):
        df = read_ck_csv(csv_path)
        if df is None:
            skipped.append(csv_path)
            continue
        frames.append(df)

    print(f"{csv_name}: merged {len(frames)} files, skipped {len(skipped)} empty")
    for s in skipped:
        print("   skipped:", s.relative_to(BATCH_ROOT))

    if not frames:
        raise SystemExit(f"No valid {csv_name} found under {BATCH_ROOT}")

    combined = pd.concat(frames, ignore_index=True)

    # numeric file id from the path (works for / and \)
    combined["file_id"] = (
        combined["file"].str.extract(r"(\d+)\.java$")[0].astype("Int64")
    )

    # exact-ID exclusion (not substring)
    combined = combined[combined["file_id"] != EXCLUDED_FILE_ID]

    before = len(combined)
    combined = combined.drop_duplicates(subset=DEDUP_KEYS[csv_name])
    print(f"   dropped {before - len(combined)} duplicate rows")

    return combined.drop(columns=["file_id"]).reset_index(drop=True), combined["file_id"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_df, class_ids = combine("class.csv")
    method_df, _ = combine("method.csv")

    class_df.to_csv(OUTPUT_DIR / "class.csv", index=False)
    method_df.to_csv(OUTPUT_DIR / "method.csv", index=False)

    print(f"\nCombined class.csv: {len(class_df)} rows -> {OUTPUT_DIR / 'class.csv'}")
    print(f"Combined method.csv: {len(method_df)} rows -> {OUTPUT_DIR / 'method.csv'}")
    print(f"Unique files in class.csv: {class_df['file'].nunique()}")
    print(f"Unique files in method.csv: {method_df['file'].nunique()}")

    # coverage vs. downloaded source files
    all_ids = {int(p.stem) for p in SOURCE_FILES_DIR.glob("*.java")} - {EXCLUDED_FILE_ID}
    covered = set(class_ids.dropna().astype(int))
    missing = sorted(all_ids - covered)
    print(f"\nSource files (minus {EXCLUDED_FILE_ID}): {len(all_ids)} | in class.csv: {len(covered)} | missing: {len(missing)}")
    print("Missing IDs (first 30):", missing[:30])
    (OUTPUT_DIR / "ck_missing_files.txt").write_text("\n".join(map(str, missing)))


if __name__ == "__main__":
    main()