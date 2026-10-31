"""
Stage 3.3: Run CK and DesigniteJava against the existing 20-snippet test sample,
both as-is (raw snippet) and wrapped (minimal synthetic class shell).
Uses the already-built jars; does not modify either tool's installation.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_DIR = PROJECT_ROOT / "data"
TEST_SAMPLE_FILE = INTERIM_DIR / "tool_test_sample.csv"
TEST_WORKDIR = INTERIM_DIR / "tool_test_workdir"

CK_JAR = PROJECT_ROOT / "tools" / "ck" / "target" / "ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar"
DESIGNITE_JAR = PROJECT_ROOT / "tools" / "DesigniteJava" / "target" / "DesigniteJava.jar"


def wrap_snippet(content: str, row_type: str) -> str:
    """
    Wrap a raw method-level snippet in a minimal synthetic class so it
    becomes a parseable compilation unit. Class-level snippets are left
    as-is; if they fail as-is we'll see that in the results.
    """
    if row_type == "function":
        return f"class SyntheticWrapper {{\n{content}\n}}\n"
    return content


def prepare_test_dirs(sample: pd.DataFrame):
    as_is_dir = TEST_WORKDIR / "as_is"
    wrapped_dir = TEST_WORKDIR / "wrapped"
    for d in (as_is_dir, wrapped_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    missing = []
    for _, row in sample.iterrows():
        snippet_path = DATA_DIR / row["snippet_path"]
        if not snippet_path.exists():
            missing.append(str(snippet_path))
            continue

        content = snippet_path.read_text(encoding="utf-8")

        (as_is_dir / f"{row['id']}.java").write_text(content, encoding="utf-8")
        wrapped_content = wrap_snippet(content, row["type"])
        (wrapped_dir / f"{row['id']}.java").write_text(wrapped_content, encoding="utf-8")

    if missing:
        print(f"WARNING: {len(missing)} snippet file(s) not found, skipped:")
        for m in missing:
            print(f"  {m}")

    return as_is_dir, wrapped_dir


def run_ck(input_dir: Path, output_dir: Path) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not CK_JAR.exists():
        return False, f"CK jar not found at {CK_JAR}"

    cmd = ["java", "-jar", str(CK_JAR), str(input_dir), "false", "0", "false", str(output_dir) + "\\"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "timed out after 120s"

    if result.returncode != 0:
        return False, (result.stderr or result.stdout)[:800]
    return True, "ok"


def run_designite(input_dir: Path, output_dir: Path) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not DESIGNITE_JAR.exists():
        return False, f"DesigniteJava jar not found at {DESIGNITE_JAR}"

    cmd = ["java", "-jar", str(DESIGNITE_JAR), "-i", str(input_dir), "-o", str(output_dir)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "timed out after 120s"

    if result.returncode != 0:
        return False, (result.stderr or result.stdout)[:800]
    return True, "ok"


def describe_output(output_dir: Path) -> str:
    if not output_dir.exists():
        return "no output dir created"
    files = list(output_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    if not files:
        return "output dir created but empty"
    return f"{len(files)} file(s): " + ", ".join(f.name for f in files[:10])


def main():
    if not TEST_SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"{TEST_SAMPLE_FILE} not found. Run select_test_sample.py first."
        )

    sample = pd.read_csv(TEST_SAMPLE_FILE)
    print(f"Loaded {len(sample)} rows from {TEST_SAMPLE_FILE}")

    as_is_dir, wrapped_dir = prepare_test_dirs(sample)

    results = []
    for mode, input_dir in [("as_is", as_is_dir), ("wrapped", wrapped_dir)]:
        ck_out = TEST_WORKDIR / f"ck_output_{mode}"
        ck_success, ck_msg = run_ck(input_dir, ck_out)
        results.append({
            "tool": "CK", "mode": mode, "success": ck_success,
            "message": ck_msg, "output_summary": describe_output(ck_out),
            "output_dir": str(ck_out),
        })

        designite_out = TEST_WORKDIR / f"designite_output_{mode}"
        designite_success, designite_msg = run_designite(input_dir, designite_out)
        results.append({
            "tool": "DesigniteJava", "mode": mode, "success": designite_success,
            "message": designite_msg, "output_summary": describe_output(designite_out),
            "output_dir": str(designite_out),
        })

    results_df = pd.DataFrame(results)
    pd.set_option("display.max_colwidth", 200)
    print("\n=== Comparison Results ===")
    print(results_df.to_string())

    results_df.to_csv(INTERIM_DIR / "tool_comparison_results.csv", index=False)
    print(f"\nSaved to {INTERIM_DIR / 'tool_comparison_results.csv'}")


if __name__ == "__main__":
    main()