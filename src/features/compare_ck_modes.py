"""
Stage 3.3 final check: for each of the 20 sample snippets, did CK produce
a class row / method row in as_is mode vs wrapped mode?
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
WORKDIR = INTERIM_DIR / "tool_test_workdir"
SAMPLE = INTERIM_DIR / "tool_test_sample.csv"


def ids_in(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "file" not in df.columns:
        raise ValueError(f"No 'file' column in {csv_path}; columns: {list(df.columns)}")
    stems = df["file"].astype(str).map(lambda p: Path(p.replace("\\", "/")).stem)
    return stems.value_counts().to_dict()


def main():
    sample = pd.read_csv(SAMPLE)
    counts = {}
    for mode in ("as_is", "wrapped"):
        out = WORKDIR / f"ck_output_{mode}"
        counts[(mode, "class")] = ids_in(out / "class.csv")
        counts[(mode, "method")] = ids_in(out / "method.csv")

    rows = []
    for _, r in sample.iterrows():
        sid = str(r["id"])
        rows.append({
            "id": sid,
            "smell": r["smell"],
            "type": r["type"],
            "as_is_class": counts[("as_is", "class")].get(sid, 0),
            "as_is_method": counts[("as_is", "method")].get(sid, 0),
            "wrapped_class": counts[("wrapped", "class")].get(sid, 0),
            "wrapped_method": counts[("wrapped", "method")].get(sid, 0),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    fn = result[result["type"] == "function"]
    cl = result[result["type"] == "class"]
    print("\n=== Coverage summary ===")
    print(f"function rows: {len(fn)} | with method metrics as_is: "
          f"{(fn['as_is_method'] > 0).sum()} | wrapped: {(fn['wrapped_method'] > 0).sum()}")
    print(f"class rows:    {len(cl)} | with class metrics as_is: "
          f"{(cl['as_is_class'] > 0).sum()} | wrapped: {(cl['wrapped_class'] > 0).sum()}")


if __name__ == "__main__":
    main()