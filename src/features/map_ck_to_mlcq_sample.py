"""
Stage 3.3 (revised) proof-of-concept: map CK full-file output back to the
20-row MLCQ test sample using class/method identity + nearest-line tie-breaking.
Uses complete downloaded source files, not snippets. Read/copy only —
does not touch existing fetched files or manifests.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
SOURCE_FILES_DIR = RAW / "source_files"
INTERIM = PROJECT_ROOT / "data" / "interim"
SAMPLE_FILE = INTERIM / "tool_test_sample.csv"

CK_JAR = PROJECT_ROOT / "tools" / "ck" / "target" / "ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar"
FULLFILE_INPUT_DIR = INTERIM / "ck_fullfile_test_input"
FULLFILE_OUTPUT_DIR = INTERIM / "ck_fullfile_test_output"


def run_ck(input_dir: Path, output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cmd = ["java", "-jar", str(CK_JAR), str(input_dir), "false", "0", "false", str(output_dir) + "\\"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"CK failed: {result.stderr[:800]}")


def file_mask(df: pd.DataFrame, file_id: int):
    """Rows whose 'file' path ends with exactly <file_id>.java (works for / and \\)."""
    return df["file"].str.contains(rf"(?:^|[\\/]){file_id}\.java$", regex=True, na=False)


def resolve_class_from_file(class_df: pd.DataFrame, file_id: int, code_name: str):
    """
    Try progressively longer dot-suffixes of code_name against CK's actual
    class names for this file, so nested classes (Outer.Inner) resolve
    correctly instead of guessing the package/class boundary.
    Returns (matched_ck_class_name, row) or (None, None).
    """
    file_classes = class_df[file_mask(class_df, file_id)]
    if len(file_classes) == 0:
        return None, None

    # CK writes nested classes as Outer$Inner; normalise to dots for comparison
    normalized = file_classes["class"].str.replace("$", ".", regex=False)

    parts = code_name.split(".")
    # try longest suffix first (most specific), down to just the last segment
    for n in range(min(4, len(parts)), 0, -1):
        candidate_suffix = ".".join(parts[-n:])
        mask = (normalized == candidate_suffix) | normalized.str.endswith("." + candidate_suffix)
        match = file_classes[mask]
        if len(match) == 1:
            return match.iloc[0]["class"], match.iloc[0]
        # 0 or >1 matches: try next suffix length; if none resolve, fall through to failure
    return None, None


def split_method_code_name(code_name: str):
    """
    Returns (class_part, method_name, param_type_hint) or (None, None, None)
    if the format isn't recognized with sufficient confidence.
    param_type_hint is the trailing parameter-type token when MLCQ appended
    one to disambiguate an overload (e.g. "isInfinite Number"), else None.
    """
    if "|" in code_name:
        return None, None, None

    if "#" in code_name:
        class_part, method_part = code_name.rsplit("#", 1)
        return class_part, method_part, None

    if " " in code_name:
        # "Class.method ParamType": exactly one space, tail is a bare type name
        left, _, tail = code_name.rpartition(" ")
        if " " in left or not tail or not tail[0].isupper() or "." in tail:
            return None, None, None
        parts = left.split(".")
        if len(parts) < 2:
            return None, None, None
        return ".".join(parts[:-1]), parts[-1], tail

    parts = code_name.split(".")
    if len(parts) < 2:
        return None, None, None
    return ".".join(parts[:-1]), parts[-1], None


def match_method(method_df: pd.DataFrame, file_id: int, class_part: str,
                 target_method: str, start_line: int, class_df: pd.DataFrame,
                 param_type_hint: str = None):
    ck_class_name, _ = resolve_class_from_file(class_df, file_id, class_part)
    if ck_class_name is None:
        return None

    candidates = method_df[file_mask(method_df, file_id) & (method_df["class"] == ck_class_name)]
    # method column looks like "name/0" or "name/1[java.lang.String]" or is a constructor
    ck_method_name = candidates["method"].str.split("/").str[0]
    candidates = candidates[ck_method_name == target_method]
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates.iloc[0]

    # multiple overloads: the param-type hint is more specific than line proximity
    if param_type_hint:
        hinted = candidates[candidates["method"].str.contains(param_type_hint, regex=False)]
        if len(hinted) == 1:
            return hinted.iloc[0]
        if len(hinted) > 1:
            candidates = hinted  # narrow, then break remaining ties by line

    # pick nearest CK 'line' to MLCQ start_line
    candidates = candidates.copy()
    candidates["line_diff"] = (candidates["line"] - start_line).abs()
    return candidates.sort_values("line_diff").iloc[0]


def main():
    sample = pd.read_csv(SAMPLE_FILE)

    if FULLFILE_INPUT_DIR.exists():
        shutil.rmtree(FULLFILE_INPUT_DIR)
    FULLFILE_INPUT_DIR.mkdir(parents=True)

    file_ids = sample["file_id"].dropna().astype(int).unique()
    copied = 0
    for fid in file_ids:
        src = SOURCE_FILES_DIR / f"{fid}.java"
        if src.exists():
            shutil.copy(src, FULLFILE_INPUT_DIR / f"{fid}.java")
            copied += 1
    print(f"Copied {copied}/{len(file_ids)} full source files into {FULLFILE_INPUT_DIR}")

    run_ck(FULLFILE_INPUT_DIR, FULLFILE_OUTPUT_DIR)

    class_df = pd.read_csv(FULLFILE_OUTPUT_DIR / "class.csv")
    method_df = pd.read_csv(FULLFILE_OUTPUT_DIR / "method.csv")

    rows_out = []
    for _, row in sample.iterrows():
        file_id = int(row["file_id"])

        if row["type"] == "class":
            ck_class, match = resolve_class_from_file(class_df, file_id, row["code_name"])
            ck_method, ck_line = None, None
            success = ck_class is not None
        else:
            class_part, method_name, param_hint = split_method_code_name(row["code_name"])
            if class_part is None:
                ck_class, ck_method, ck_line = None, None, None
                success = False
            else:
                match = match_method(method_df, file_id, class_part, method_name,
                                     row["start_line"], class_df, param_hint)
                ck_class = match["class"] if match is not None else None
                ck_method = match["method"] if match is not None else None
                ck_line = match["line"] if match is not None else None
                success = match is not None

        rows_out.append({
            "id": row["id"], "smell": row["smell"], "type": row["type"],
            "code_name": row["code_name"], "start_line": row["start_line"],
            "ck_class": ck_class, "ck_method": ck_method, "ck_line": ck_line,
            "mapped": success,
        })

    result = pd.DataFrame(rows_out)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== Mapping table (all 20 rows) ===")
    print(result.to_string(index=False))

    for t in ("class", "function"):
        subset = result[result["type"] == t]
        if len(subset):
            print(f"\n{t} rows: {len(subset)} | mapped: {subset['mapped'].sum()} "
                  f"({100 * subset['mapped'].mean():.0f}%)")


if __name__ == "__main__":
    main()