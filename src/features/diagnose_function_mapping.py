import sys
import difflib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
from map_ck_to_mlcq_sample import resolve_class_from_file, split_method_code_name

feat = pd.read_csv(ROOT / "data/processed/features_full_unfiltered.csv")
cls = pd.read_csv(ROOT / "data/interim/ck_full_combined/class.csv")
met = pd.read_csv(ROOT / "data/interim/ck_full_combined/method.csv")

# optional: bring in start_line if the labeled file exists and shares an "id" column
if "start_line" not in feat.columns:
    hits = list((ROOT / "data").rglob("labeled_rows_with_files.csv"))
    if hits:
        lab = pd.read_csv(hits[0])
        if "id" in lab.columns and "start_line" in lab.columns and "id" in feat.columns:
            feat = feat.merge(lab[["id", "start_line"]].drop_duplicates("id"), on="id", how="left")
            print("joined start_line from", hits[0].name)
    if "start_line" not in feat.columns:
        feat["start_line"] = pd.NA
        print("start_line not available; printing without it")

for df in (cls, met):
    df["fid"] = df["file"].str.extract(r"(\d+)\.java$")[0].astype("Int64")
    df["mname"] = df["method"].str.split("/").str[0] if "method" in df.columns else None
cls_by = {k: g for k, g in cls.groupby("fid")}
met_by = {k: g for k, g in met.groupby("fid")}

un = feat[(feat["type"] == "function") & (~feat["mapped"].astype(bool))].copy()
print("unmapped function rows:", len(un))


def diagnose(row):
    fid = int(row["file_id"])
    cn = row["code_name"]
    c, m = cls_by.get(fid), met_by.get(fid)
    if c is None:
        return "file_not_in_ck", None, None
    cp, mn, hint = split_method_code_name(cn)
    if cp is None:
        if "|" in cn: return "rejected_pipe", None, None
        if " " in cn: return "rejected_space", None, None
        return "rejected_no_dot", None, None
    ck_cls, _ = resolve_class_from_file(c, fid, cp)
    if ck_cls is None:
        return "class_unresolved", None, mn
    if m is None:
        return "class_has_no_methods", ck_cls, mn
    if mn in set(m[m["class"] == ck_cls]["mname"]):
        return "would_map_with_sample_logic", ck_cls, mn
    if mn in set(m["mname"]):
        return "method_in_different_class", ck_cls, mn
    if mn == cp.split(".")[-1]:
        return "constructor_name", ck_cls, mn
    return "method_not_found", ck_cls, mn


res = un.apply(diagnose, axis=1, result_type="expand")
un["bucket"], un["ck_cls"], un["target_method"] = res[0], res[1], res[2]
print(un["bucket"].value_counts().to_string())

# ---------------------------------------------------------------- pipe bucket
pipe = un[un["bucket"] == "rejected_pipe"].copy()


def pipe_shape(cn):
    head = cn.split(" ")[0]
    tail = cn.split(" ", 1)[1] if " " in cn else ""
    cand = head.split("#")[-1].split(".")[-1]
    return pd.Series({
        "n_pipes": cn.count("|"),
        "has_space": " " in cn,
        "pipe_in_head": "|" in head,      # before first space
        "pipe_in_tail": "|" in tail,      # after first space
        "has_hash": "#" in cn,
        "cand_method": cand,
    })


pipe = pd.concat([pipe, pipe["code_name"].apply(pipe_shape)], axis=1)
pipe["cand_in_file"] = pipe.apply(
    lambda r: r["cand_method"] in set(met_by[int(r["file_id"])]["mname"])
    if int(r["file_id"]) in met_by else False, axis=1)

print("\n=== PIPE BUCKET: shape summary ===")
print(pipe.groupby(["has_space", "pipe_in_head", "pipe_in_tail", "has_hash"]).size()
      .rename("rows").to_string())
print("\nn_pipes distribution:\n", pipe["n_pipes"].value_counts().sort_index().to_string())
print(f"\nfirst-token method name exists somewhere in the same CK file: "
      f"{pipe['cand_in_file'].sum()}/{len(pipe)}")

print("\n=== PIPE BUCKET: random examples with the CK methods of that file ===")
for _, r in pipe.sample(min(15, len(pipe)), random_state=0).iterrows():
    ms = met_by.get(int(r["file_id"]))
    ck = [] if ms is None else ms[ms["mname"] == r["cand_method"]]["method"].tolist()[:4]
    print(f"\n  file {r['file_id']} line {r['start_line']}  [{r['smell']}]")
    print(f"    code_name : {r['code_name']}")
    print(f"    candidate : {r['cand_method']}")
    print(f"    CK methods with that name: {ck}")

# ------------------------------------------------------- method_not_found bucket
mnf = un[un["bucket"] == "method_not_found"].copy()


def nf_reason(r):
    t = r["target_method"]
    ms = met_by[int(r["file_id"])]
    in_cls = ms[ms["class"] == r["ck_cls"]]["mname"]
    names = set(in_cls)
    if t is None:
        return "no_target"
    if not names:
        return "class_has_zero_methods_in_CK"
    if t.lower() in {n.lower() for n in names}:
        return "case_difference"
    if any(ch in t for ch in "(<>[]"):
        return "target_has_symbols"
    if difflib.get_close_matches(t, list(names), n=1, cutoff=0.8):
        return "close_name_exists"
    return "no_similar_name_in_class"


mnf["reason"] = mnf.apply(nf_reason, axis=1)
print("\n=== METHOD_NOT_FOUND: reason summary ===")
print(mnf["reason"].value_counts().to_string())

print("\n=== METHOD_NOT_FOUND: random examples ===")
for _, r in mnf.sample(min(20, len(mnf)), random_state=0).iterrows():
    ms = met_by[int(r["file_id"])]
    in_cls = ms[ms["class"] == r["ck_cls"]]
    names = sorted(set(in_cls["mname"]))
    close = difflib.get_close_matches(r["target_method"], names, n=3, cutoff=0.5)
    print(f"\n  file {r['file_id']} line {r['start_line']}  [{r['smell']}]  reason={r['reason']}")
    print(f"    code_name     : {r['code_name']}")
    print(f"    resolved class: {r['ck_cls']}   target method: {r['target_method']}")
    print(f"    closest CK names: {close}")
    print(f"    CK method count in class: {len(names)}  e.g. {names[:6]}")

out = ROOT / "data/interim/function_mapping_diagnostics.csv"
keep = ["id", "file_id", "smell", "code_name", "start_line", "bucket", "ck_cls", "target_method"]
un[[c for c in keep if c in un.columns]].to_csv(out, index=False)
print("\nwrote", out)