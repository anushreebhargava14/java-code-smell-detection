"""
Stage 3.1 & 3.2: Join MLCQ labels to fetched files, extract referenced snippets.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
SAMPLES_FILE = RAW_DATA_DIR / "MLCQCodeSmellSamples.csv"
MANIFEST_FILE = RAW_DATA_DIR / "fetch_manifest.csv"
SOURCE_FILES_DIR = RAW_DATA_DIR / "source_files"
SNIPPETS_DIR = INTERIM_DIR / "snippets"
JOINED_OUTPUT = INTERIM_DIR / "labeled_rows_with_files.csv"


def extract_snippet(file_id: int, start_line: int, end_line: int) -> str | None:
    source_path = SOURCE_FILES_DIR / f"{file_id}.java"
    if not source_path.exists():
        return None

    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # MLCQ line numbers are 1-indexed and inclusive.
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))

    if start_idx >= len(lines):
        return None

    return "\n".join(lines[start_idx:end_idx])


def main():
    df = pd.read_csv(SAMPLES_FILE, sep=";")
    manifest = pd.read_csv(MANIFEST_FILE)

    # Join on the same key used to build the manifest
    merged = df.merge(
        manifest[["repository", "commit_hash", "path", "file_id", "status"]],
        on=["repository", "commit_hash", "path"],
        how="left",
    )

    print(f"Total MLCQ rows: {len(df)}")
    print(f"Rows after join: {len(merged)}")

    missing_join = merged["file_id"].isna().sum()
    print(f"Rows with no manifest match at all: {missing_join}")

    successful = merged[merged["status"] == "success"].copy()
    print(f"Rows with successfully fetched file: {len(successful)}")
    print(f"Rows dropped (failed/missing fetch): {len(merged) - len(successful)}")

    SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)

    snippet_paths = []
    empty_snippets = 0

    for _, row in successful.iterrows():
        file_id = int(row["file_id"])
        row_id = row["id"]
        snippet = extract_snippet(file_id, int(row["start_line"]), int(row["end_line"]))

        if snippet is None or snippet.strip() == "":
            empty_snippets += 1
            snippet_paths.append(None)
            continue

        snippet_path = SNIPPETS_DIR / f"{row_id}.java"
        snippet_path.write_text(snippet, encoding="utf-8")
        snippet_paths.append(str(snippet_path.relative_to(INTERIM_DIR.parents[0])))

    successful["snippet_path"] = snippet_paths
    successful["label"] = (successful["severity"] != "none").astype(int)

    successful.to_csv(JOINED_OUTPUT, index=False)

    print(f"\nRows with empty/failed snippet extraction: {empty_snippets}")
    print(f"Final joined+labeled table saved to: {JOINED_OUTPUT}")
    print(f"Final usable rows (non-empty snippet): {len(successful) - empty_snippets}")


if __name__ == "__main__":
    main()