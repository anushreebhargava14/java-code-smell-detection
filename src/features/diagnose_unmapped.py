"""
Stage 3.4 diagnostic (READ-ONLY). Explains the remaining unmapped rows and the
large line_diff rows using the feature table, CK outputs and the original source files.
Writes only new CSVs to data/interim/diagnostics/.
"""

import difflib
import re

import pandas as pd

import map_ck_to_mlcq_full as m   # reuses loaded CK tables + helpers; main() is not run

ROOT = m.ROOT
SOURCE_DIR = ROOT / "data" / "raw" / "source_files"
DIAG_DIR = ROOT / "data" / "interim" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

feat = pd.read_csv(ROOT / "data" / "processed" / "features_full_unfiltered.csv")
cls_by, met_by = m.cls_by, m.met_by
resolve = m.resolve_class_from_file

_cache = {}


def src_lines(fid):
    if fid not in _cache:
        p = SOURCE_DIR / f"{fid}.java"
        _cache[fid] = (p.read_text(encoding="utf-8", errors="replace")
                       .replace("\r\n", "\n").split("\n")) if p.exists() else None
    return _cache[fid]


def name_near_line(fid, name, line, before=1, after=3):
    lines = src_lines(fid)
    if lines is None or pd.isna(line):
        return None
    seg = "\n".join(lines[max(int(line) - 1 - before, 0): int(line) + after])
    return bool(re.search(rf"\b{re.escape(name)}\s*\(", seg))


def simple_of(class_part):
    return re.split(r"[.$]", class_part)[-1]


def class_decl_line(fid, simple):
    lines = src_lines(fid)
    if lines is None:
        return None
    rx = re.compile(rf"\b(?:class|interface|enum)\s+{re.escape(simple)}\b")
    for i, ln in enumerate(lines, start=1):
        if rx.search(ln):
            return i
    return None


def nearest_in_class(fid, ck_cls, name, start):
    mdf = met_by.get(fid)
    if mdf is None:
        return None
    cand = mdf[(mdf["class"] == ck_cls) & (mdf["mname"] == name)]
    if cand.empty:
        return None
    d = (cand["line"] - start).abs()
    i = d.idxmin()
    return cand.loc[i, "method"], int(cand.loc[i, "line"]), int(d.loc[i])


def short(s, n=70):
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def show_examples(df, cols, per_group, group_col):
    for g, sub in df.groupby(group_col):
        print(f"\n  [{g}] ({len(sub)} rows) examples:")
        print(sub[cols].head(per_group).to_string(index=False))


# ---------------------------------------------------------------- 1. name_not_in_class
def diag_name_not_in_class(r):
    fid, start = int(r["file_id"]), r["start_line"]
    cp, name, _ = m.parse_code_name(r["code_name"])
    ck_cls, _ = resolve(cls_by[fid], fid, cp)
    mdf = met_by[fid]
    in_cls = mdf[mdf["class"] == ck_cls]
    near = difflib.get_close_matches(name.lower(), [x.lower() for x in in_cls["mname"].unique()],
                                     n=3, cutoff=0.75)
    others = mdf[(mdf["mname"] == name) & (mdf["class"] != ck_cls)]
    others_close = others[(others["line"] - start).abs() <= 2]
    anchors = in_cls[(in_cls["line"] - start).abs() <= 1]
    has_name = name_near_line(fid, name, start)

    if others_close["class"].nunique() == 1:
        verdict, conf = "same_name_in_other_class_at_line", "high"
    elif others["class"].nunique() == 1:
        verdict, conf = "same_name_in_other_class_far_line", "low"
    elif others["class"].nunique() > 1:
        verdict, conf = "same_name_in_several_other_classes", "low"
    elif len(anchors) and set(anchors["mname"].str.lower()) & set(near):
        verdict, conf = "near_name_at_same_line", "medium"
    elif len(anchors):
        verdict, conf = "different_name_at_same_line", "low"
    elif has_name:
        verdict, conf = "ck_has_no_such_method(source_has_it)", "none"
    else:
        verdict, conf = "name_absent_in_source_at_line", "none"

    return {
        "id": r["id"], "smell": r["smell"], "file_id": fid, "code_name": r["code_name"],
        "start_line": start, "ck_class": ck_cls, "verdict": verdict, "confidence": conf,
        "near_names": ";".join(near),
        "other_class_same_name": ";".join(f"{c}@{l}" for c, l in zip(others["class"], others["line"]))[:200],
        "line_anchor_methods": ";".join(f"{x}@{l}" for x, l in zip(anchors["method"], anchors["line"]))[:200],
        "source_site_has_name": has_name,
    }


# ------------------------------------------------- 2/3. class_unresolved (function + class)
def diag_class_unresolved(r):
    fid, start = int(r["file_id"]), r["start_line"]
    if r["type"] == "function":
        cp, name, _ = m.parse_code_name(r["code_name"])
    else:
        cp, name = r["code_name"], None
    simple = simple_of(cp)
    fc = cls_by.get(fid)
    ck_list = [] if fc is None else list(fc["class"])

    t_dollar = resolve(fc, fid, cp.replace("$", ".")) [0] if fc is not None else None
    t_simple = None
    if fc is not None:
        norm = fc["class"].str.replace("$", ".", regex=False).str.split(".").str[-1]
        hits = fc[norm == simple]["class"].unique()
        t_simple = hits[0] if len(hits) == 1 else None
    decl = class_decl_line(fid, simple)

    target, how = (t_dollar, "dollar_normalization") if t_dollar else \
                  (t_simple, "simple_name_unique") if t_simple else (None, None)
    if target is None:
        verdict = "class_not_in_source_file" if decl is None else "class_in_source_not_in_ck"
        conf, evidence = "none", ""
    else:
        verdict = how
        if r["type"] == "function":
            hit = nearest_in_class(fid, target, name, start)
            if hit is None:
                conf, evidence = "low", "target class found but method not in it"
            else:
                conf = "high" if hit[2] <= 2 else "medium"
                evidence = f"{hit[0]}@{hit[1]} (line diff {hit[2]})"
        else:
            if decl is not None and abs(decl - start) <= 2:
                conf = "high"
            else:
                conf = "medium"
            evidence = f"decl_line={decl} vs start_line={start}"

    return {
        "id": r["id"], "type": r["type"], "smell": r["smell"], "file_id": fid,
        "code_name": r["code_name"], "start_line": start, "verdict": verdict,
        "confidence": conf, "target_ck_class": target, "evidence": evidence,
        "class_decl_line_in_source": decl, "ck_classes_in_file": ";".join(ck_list[:6]),
    }


# --------------------------------------------------------------- 4. large line_diff
def diag_big_line_diff(r):
    fid, start, end = int(r["file_id"]), r["start_line"], r["end_line"]
    ck_cls, ck_line = r["ck_class"], int(r["ck_line"])
    name = str(r["ck_method"]).split("/")[0]
    mdf = met_by[fid]
    cand = mdf[(mdf["class"] == ck_cls) & (mdf["mname"] == name)]
    d = (cand["line"] - start).abs()
    best_diff = int(d.min())
    chosen_diff = abs(ck_line - start)
    better = bool((d <= 2).any() and best_diff < chosen_diff)
    in_range = bool(start <= ck_line <= end)
    ck_site = name_near_line(fid, name, ck_line)
    mlcq_site = name_near_line(fid, name, start)

    if better:
        verdict = "likely_wrong_overload"
    elif in_range:
        verdict = "plausible_ck_line_inside_mlcq_range"
    elif chosen_diff == best_diff and ck_site:
        verdict = "plausible_nearest_same_name_candidate"
    else:
        verdict = "unverified"

    return {
        "id": r["id"], "smell": r["smell"], "match_source": r["match_source"],
        "code_name": r["code_name"], "start_line": start, "end_line": end,
        "ck_class": ck_cls, "ck_method": r["ck_method"], "ck_line": ck_line,
        "line_diff": chosen_diff, "n_same_name_candidates": len(cand),
        "nearest_candidate_diff": best_diff, "ck_line_in_mlcq_range": in_range,
        "source_has_name_at_mlcq_line": mlcq_site, "source_has_name_at_ck_line": ck_site,
        "verdict": verdict,
    }


def main():
    unm = feat[~feat["mapped"]]
    print("=== Reconciliation: unmapped rows by (type, match_source) ===")
    print(unm.groupby(["type", "match_source"]).size().to_string())

    # ---- 1
    rows1 = unm[(unm["type"] == "function") & (unm["match_source"] == "name_not_in_class")]
    d1 = pd.DataFrame([diag_name_not_in_class(r) for _, r in rows1.iterrows()])
    d1.to_csv(DIAG_DIR / "1_name_not_in_class.csv", index=False)
    print(f"\n=== 1. name_not_in_class (function rows): {len(d1)} ===")
    print(d1.groupby(["verdict", "confidence"]).size().to_string())
    d1s = d1.assign(code_name=d1["code_name"].map(short))
    show_examples(d1s, ["id", "code_name", "start_line", "other_class_same_name",
                        "line_anchor_methods", "near_names"], 4, "verdict")

    # ---- 2 + 3
    rows2 = unm[(unm["type"] == "function") & (unm["match_source"] == "class_unresolved")]
    rows3 = unm[(unm["type"] == "class") & (unm["match_source"] == "class_unresolved")]
    d2 = pd.DataFrame([diag_class_unresolved(r) for _, r in rows2.iterrows()])
    d3 = pd.DataFrame([diag_class_unresolved(r) for _, r in rows3.iterrows()])
    d2.to_csv(DIAG_DIR / "2_class_unresolved_function.csv", index=False)
    d3.to_csv(DIAG_DIR / "3_class_unresolved_class.csv", index=False)
    for title, d in (("2. class_unresolved (function rows)", d2), ("3. class_unresolved (class rows)", d3)):
        print(f"\n=== {title}: {len(d)} ===")
        if len(d):
            print(d.groupby(["verdict", "confidence"]).size().to_string())
            ds = d.assign(code_name=d["code_name"].map(short))
            show_examples(ds, ["id", "code_name", "target_ck_class", "evidence",
                               "class_decl_line_in_source"], 4, "verdict")

    # ---- 4
    big = feat[(feat["type"] == "function") & feat["mapped"] & (feat["line_diff"] > 50)]
    d4 = pd.DataFrame([diag_big_line_diff(r) for _, r in big.iterrows()])
    d4.to_csv(DIAG_DIR / "4_large_line_diff.csv", index=False)
    print(f"\n=== 4. mapped function rows with line_diff > 50: {len(d4)} ===")
    print(pd.crosstab(d4["match_source"], d4["verdict"]).to_string())
    d4["bin"] = pd.cut(d4["line_diff"], [50, 100, 500, 100000], labels=["51-100", "101-500", ">500"])
    print("\nline_diff bins by match_source:")
    print(pd.crosstab(d4["match_source"], d4["bin"]).to_string())
    d4s = d4.assign(code_name=d4["code_name"].map(short))
    for src, sub in d4s.groupby("match_source"):
        print(f"\n  [{src}] examples (largest diffs first):")
        print(sub.sort_values("line_diff", ascending=False)
              [["id", "code_name", "start_line", "end_line", "ck_line", "line_diff",
                "n_same_name_candidates", "verdict"]].head(5).to_string(index=False))

    # ---- 5
    exc = feat[feat["match_source"] == "excluded_ck_oom"]
    f1760 = feat[feat["file_id"] == 1760]
    print("\n=== 5. excluded_ck_oom check ===")
    print(f"excluded rows: {len(exc)} | distinct file_ids: {sorted(exc['file_id'].unique())}")
    print(f"types: {exc['type'].value_counts().to_dict()} | smells: {exc['smell'].value_counts().to_dict()}")
    print(f"all rows with file_id 1760: {len(f1760)} | of which excluded: {(f1760['match_source'] == 'excluded_ck_oom').sum()}")
    print(f"1760.java in source_files: {(SOURCE_DIR / '1760.java').exists()} | "
          f"in combined CK class/method output: {1760 in cls_by}/{1760 in met_by}")
    print(f"exactly the known exception: "
          f"{set(exc['file_id'].unique()) == {1760} and len(exc) == len(f1760) == 12}")

    print(f"\nCSVs written to {DIAG_DIR}")


if __name__ == "__main__":
    main()