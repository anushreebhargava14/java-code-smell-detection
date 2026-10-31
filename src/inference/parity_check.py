"""
Stage 7 parity check: does the inference path reproduce the training-time features/probabilities?
Read-only: runs inference on original source files and compares to the stored dataset rows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "inference"))
import predict as P  # noqa: E402

SEED, PER_SMELL, TOL = 42, 8, 1e-3
DATA = ROOT / "data" / "processed" / "dataset"
SRC = ROOT / "data" / "raw" / "source_files"

rep = pd.read_csv(ROOT / "data" / "processed" / "features_full_repaired.csv",
                  usecols=["smell", "file_id", "code_name", "usable", "ck_class", "ck_method"],
                  low_memory=False)
rep = rep[rep["usable"].astype(str) == "True"]
models = P.load_models()

# ---- pick held-out test rows: half positive, half negative, per smell
picks = []
for smell in P.SMELL_LEVEL:
    t = pd.read_csv(DATA / f"{smell}.csv")
    t = t[t["split"] == "test"]
    pos = t[t["label"] == 1].sample(PER_SMELL // 2, random_state=SEED)
    neg = t[t["label"] == 0].sample(PER_SMELL - len(pos), random_state=SEED)
    picks += [(smell, r) for _, r in pd.concat([pos, neg]).iterrows()]

cache = {}


def infer(file_id):
    if file_id not in cache:
        text = (SRC / f"{file_id}.java").read_text(encoding="utf-8", errors="replace")
        cache[file_id] = (text, P.analyze_source(text))
    return cache[file_id]


def find(res, smell, ck_class, ck_method):
    for s in res["scores"]:
        if s["smell"] == smell and s["ck_class"] == ck_class:
            if (s["ck_method"] is None and pd.isna(ck_method)) or s["ck_method"] == ck_method:
                return s
    return None


rows, diagnose = [], []
for smell, r in picks:
    art, feats = models[smell], models[smell]["features"]
    fid = int(r["file_id"])
    j = rep[(rep["smell"] == smell.replace("_", " ")) & (rep["file_id"] == fid)
            & (rep["code_name"] == r["code_name"])]
    base = {"smell": smell, "file_id": fid, "label": int(r["label"]), "entity": str(r["code_name"])[-45:]}
    if j.empty:
        rows.append({**base, "status": "NO JOIN"})
        continue
    ck_class, ck_method = j.iloc[0]["ck_class"], j.iloc[0]["ck_method"]
    p_stored = float(art["model"].predict_proba(r[feats].astype(float).to_numpy().reshape(1, -1))[0, 1])
    try:
        text, res = infer(fid)
    except Exception as e:  # noqa: BLE001
        rows.append({**base, "p_stored": p_stored, "status": f"ERROR {e}"})
        continue
    if not res["ok"]:
        rows.append({**base, "p_stored": p_stored, "status": f"INFERENCE ERROR: {res['error'][:40]}"})
        continue
    s = find(res, smell, ck_class, ck_method)
    if s is None:
        rows.append({**base, "p_stored": p_stored, "status": "ENTITY NOT FOUND"})
        diagnose.append((smell, r, ck_class, ck_method, text, "not_found"))
        continue
    diff = abs(s["probability"] - p_stored)
    status = "match" if diff <= TOL else "DIFF"
    rows.append({**base, "p_stored": round(p_stored, 4), "p_infer": s["probability"],
                 "abs_diff": round(diff, 5), "status": status})
    if status == "DIFF":
        diagnose.append((smell, r, ck_class, ck_method, text, "diff"))

out = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(out.to_string(index=False))

found = out[out["status"].isin(["match", "DIFF"])]
print("\n=== Summary ===")
print(f"rows checked: {len(out)} | match (|dp| <= {TOL}): {int((out['status'] == 'match').sum())} | "
      f"DIFF: {int((out['status'] == 'DIFF').sum())} | other: {int((~out['status'].isin(['match', 'DIFF'])).sum())}")
if len(found):
    print(f"abs probability difference: median {found['abs_diff'].median():.6f}, max {found['abs_diff'].max():.5f}")
    print(f"identical to 6 decimals: {int((found['abs_diff'] < 1e-6).sum())} of {len(found)}")

# ---- diagnosis of mismatches: which features differ?
for smell, r, ck_class, ck_method, text, kind in diagnose:
    feats = models[smell]["features"]
    level = P.SMELL_LEVEL[smell]
    print(f"\n--- {smell} | file {int(r['file_id'])} | {kind} | {ck_class} {'' if pd.isna(ck_method) else ck_method}")
    classes, methods, _ = P.run_ck(text)
    df = (classes if level == "class" else methods).reset_index(drop=True)
    if kind == "not_found":
        print("  CK classes in this file:", list(classes["class"])[:6] if "class" in classes else "none")
        continue
    mask = df["class"] == ck_class
    if level == "method":
        mask &= df["method"] == ck_method
    X = P.feature_matrix(df[mask].reset_index(drop=True), feats, [])[0]
    stored = r[feats].astype(float).to_numpy()
    bad = [(f, a, b) for f, a, b in zip(feats, stored, X)
           if not (np.isnan(b) and np.isnan(a)) and abs(a - b) > 1e-9]
    print("  features that differ (stored vs inference):",
          [(f, a, b) for f, a, b in bad[:8]] or "none")