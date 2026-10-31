
"""
Stage 3.4 (v2): Full-dataset MLCQ -> CK mapping.
Class rows: validated resolve_class_from_file (unchanged logic).
Function rows: parse_code_name / match_v2 copied verbatim from recover_function_mapping.py.
All CK metrics kept, prefixed 'ck_'. File 1760 is the documented CK OOM exception.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
from map_ck_to_mlcq_sample import resolve_class_from_file

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
EXCLUDED_FILE_ID = 1760

labeled = pd.read_csv(INTERIM / "labeled_rows_with_files.csv")
cls = pd.read_csv(INTERIM / "ck_full_combined" / "class.csv")
met = pd.read_csv(INTERIM / "ck_full_combined" / "method.csv")

for df in (cls, met):
    df["fid"] = df["file"].str.extract(r"(\d+)\.java$")[0].astype("Int64")

met["mname"] = met["method"].str.split("/").str[0]

cls_by = {k: g for k, g in cls.groupby("fid")}
met_by = {k: g for k, g in met.groupby("fid")}


# ---- copied verbatim from recover_function_mapping.py ----

def split_top(s, sep):
    """Split on sep only outside <...> so generics stay intact."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def parse_code_name(cn: str):
    """'Class#method T1|T2' or 'Class.method T1|T2' -> (class_part, method, [types] or None)."""
    head, _, tail = cn.strip().partition(" ")
    params = split_top(tail, "|") if tail.strip() else None

    if "#" in head:
        c, m = head.rsplit("#", 1)
    elif "." in head:
        c, m = head.rsplit(".", 1)
    else:
        return None, None, None

    return c, m, params


def norm_type(t: str):
    t = t.replace("...", "[]").replace("$", ".")
    t = re.sub(r"(?:\w+\.)+(?=\w)", "", t)
    return re.sub(r"\s+", "", t)


def ck_params(method: str):
    if "[" not in method:
        return []
    return split_top(method[method.index("[") + 1: method.rindex("]")], ",")


def match_v2(fid, class_part, name, start_line, params):
    c, m = cls_by.get(fid), met_by.get(fid)

    if c is None or m is None:
        return None, "file_not_in_ck"

    ck_cls, _ = resolve_class_from_file(c, fid, class_part)

    if ck_cls is None:
        return None, "class_unresolved"

    cand = m[(m["class"] == ck_cls) & (m["mname"] == name)]

    if cand.empty:
        return None, "name_not_in_class"

    if len(cand) == 1:
        return cand.iloc[0], "unique_name"

    def nearest(df):
        if pd.isna(start_line):
            return df.iloc[0]
        return (
            df.assign(d=(df["line"] - start_line).abs())
            .sort_values("d")
            .iloc[0]
        )

    if params is not None:
        want = [norm_type(p) for p in params]

        scores = cand["method"].apply(
            lambda s: (
                sum(
                    a == norm_type(b)
                    for a, b in zip(want, ck_params(s))
                )
                if len(ck_params(s)) == len(want)
                else -1
            )
        )

        exact = cand[scores == len(want)]
        if len(exact):
            return nearest(exact), "param_exact"

        arity = cand[scores >= 0]
        if len(arity):
            return nearest(
                arity[arity.index.isin(scores[scores == scores.max()].index)]
            ), "param_arity"

    return nearest(cand), "line_only"

# ---- end verbatim section ----


HELPER_COLS = {"fid", "mname", "d"}


def ck_columns(series):
    return {
        f"ck_{k}": v
        for k, v in series.items()
        if k not in HELPER_COLS
    }


def main():
    print(f"Total labeled rows: {len(labeled)}", flush=True)

    rows = []
    legacy_disagreements = 0
    legacy_checked = 0

    for row_i, (_, r) in enumerate(labeled.iterrows()):

        # Progress reporting: every 1,000 rows
        if row_i % 1000 == 0:
            print(
                f"  processed {row_i} / {len(labeled)} rows...",
                flush=True
            )

        fid = int(r["file_id"])

        out = {
            "id": r["id"],
            "smell": r["smell"],
            "type": r["type"],
            "severity": r["severity"],
            "label": r["label"],
            "code_name": r["code_name"],
            "file_id": fid,
            "start_line": r["start_line"],
            "end_line": r["end_line"],
        }

        hit, line_diff = None, None

        if fid == EXCLUDED_FILE_ID:
            src = "excluded_ck_oom"

        elif r["type"] == "class":

            if fid not in cls_by:
                src = "file_not_in_ck"

            else:
                ck_cls, hit = resolve_class_from_file(
                    cls_by[fid], fid, r["code_name"]
                )

                src = (
                    "class_resolved"
                    if ck_cls is not None
                    else "class_unresolved"
                )

                # Sampled regression check against the legacy whole-table call.
                # This diagnostic does not affect mapping results.
                if row_i % 25 == 0:
                    legacy_checked += 1

                    legacy_cls, _ = resolve_class_from_file(
                        cls, fid, r["code_name"]
                    )

                    if legacy_cls != ck_cls:
                        legacy_disagreements += 1

        else:
            cp, mn, params = parse_code_name(r["code_name"])

            if cp is None:
                src = "unparsable"

            else:
                hit, src = match_v2(
                    fid, cp, mn, r["start_line"], params
                )

            if hit is not None and pd.notna(r["start_line"]):
                line_diff = abs(hit["line"] - r["start_line"])

        out["match_source"] = src
        out["mapped"] = hit is not None
        out["line_diff"] = line_diff

        if hit is not None:
            out.update(ck_columns(hit))

        rows.append(out)

    result = pd.DataFrame(rows)

    PROCESSED.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        PROCESSED / "features_full_unfiltered.csv",
        index=False
    )

    print(
        f"\nOverall mapping: {result['mapped'].sum()}/{len(result)} "
        f"({100 * result['mapped'].mean():.1f}%)"
    )

    print("\n--- By type ---")
    print(
        result.groupby("type")["mapped"]
        .agg(["sum", "count", "mean"])
        .round(3)
    )

    print("\n--- By smell ---")
    print(
        result.groupby("smell")["mapped"]
        .agg(["sum", "count", "mean"])
        .round(3)
    )

    print("\n--- match_source by type ---")
    print(pd.crosstab(result["match_source"], result["type"]))

    print("\n--- Unresolved rows by type / bucket ---")
    print(
        result[~result["mapped"]]
        .groupby(["type", "match_source"])
        .size()
    )

    print(
        f"\nLegacy whole-table check: sampled {legacy_checked} class rows "
        f"(every 25th row); disagreements with exact-file call: "
        f"{legacy_disagreements}"
    )

    fn = result[result["type"] == "function"]

    print("\n--- line_diff (function rows) by match_source ---")
    print(fn.groupby("match_source")["line_diff"].describe().round(1))

    print(
        "function rows with line_diff > 50:",
        int((fn["line_diff"] > 50).sum())
    )

    unmapped_cls = result[
        (result["type"] == "class") & (~result["mapped"])
    ]

    print(
        f"\nUnmapped class rows: {len(unmapped_cls)}; "
        f"first 10 code_names:"
    )

    print(unmapped_cls["code_name"].head(10).to_string(index=False))

    print(
        f"\nSaved: {PROCESSED / 'features_full_unfiltered.csv'}"
    )


if __name__ == "__main__":
    main()
