import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
from map_ck_to_mlcq_sample import resolve_class_from_file, file_mask

diag = pd.read_csv(ROOT / "data/interim/function_mapping_diagnostics.csv")
cls = pd.read_csv(ROOT / "data/interim/ck_full_combined/class.csv")
met = pd.read_csv(ROOT / "data/interim/ck_full_combined/method.csv")

for df in (cls, met):
    df["fid"] = df["file"].str.extract(r"(\d+)\.java$")[0].astype("Int64")
met["mname"] = met["method"].str.split("/").str[0]
cls_by = {k: g for k, g in cls.groupby("fid")}
met_by = {k: g for k, g in met.groupby("fid")}


def split_top(s, sep):
    """Split on sep only outside <...> so generics stay intact."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "<": depth += 1
        elif ch == ">": depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur)); cur = []
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
    t = re.sub(r"(?:\w+\.)+(?=\w)", "", t)   # drop package / outer-class prefixes
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
        return df.assign(d=(df["line"] - start_line).abs()).sort_values("d").iloc[0]

    if params is not None:
        want = [norm_type(p) for p in params]
        scores = cand["method"].apply(
            lambda s: (sum(a == norm_type(b) for a, b in zip(want, ck_params(s)))
                       if len(ck_params(s)) == len(want) else -1))
        exact = cand[scores == len(want)]
        if len(exact):
            return nearest(exact), "param_exact"
        arity = cand[scores >= 0]
        if len(arity):
            return nearest(arity[arity.index.isin(scores[scores == scores.max()].index)]), "param_arity"
    return nearest(cand), "line_only"


rows = []
for _, r in diag[diag["bucket"] != "file_not_in_ck"].iterrows():
    cp, mn, params = parse_code_name(r["code_name"])
    if cp is None:
        rows.append({"id": r["id"], "old_bucket": r["bucket"], "match_source": "unparsable"})
        continue
    hit, src = match_v2(int(r["file_id"]), cp, mn, r.get("start_line"), params)
    rows.append({
        "id": r["id"], "old_bucket": r["bucket"], "match_source": src,
        "ck_class": None if hit is None else hit["class"],
        "ck_method": None if hit is None else hit["method"],
        "ck_line": None if hit is None else hit["line"],
        "start_line": r.get("start_line"),
    })

out = pd.DataFrame(rows)
out["mapped"] = out["ck_method"].notna()
print("recovered:", int(out["mapped"].sum()), "of", len(out))
print("\nby match_source:\n", out["match_source"].value_counts().to_string())
print("\nby old bucket (mapped / total):\n",
      out.groupby("old_bucket")["mapped"].agg(["sum", "count"]).to_string())

# sanity: how far apart are MLCQ and CK lines on the exact-parameter matches?
ex = out[out["match_source"] == "param_exact"].dropna(subset=["start_line", "ck_line"])
print("\n|start_line - ck_line| on param_exact:\n",
      (ex["start_line"] - ex["ck_line"]).abs().describe().round(1).to_string())

out.to_csv(ROOT / "data/interim/function_mapping_recovered.csv", index=False)