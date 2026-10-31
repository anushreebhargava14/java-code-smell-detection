"""
Stage 4 / Step 1: one row per code location (smell, sample_id), majority-vote label
(ties dropped), per-smell feature tables, repository-aware train/test split plus
5-fold grouped CV folds inside train. Nothing is trained here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
NON_FEATURES = {"ck_file", "ck_class", "ck_method", "ck_line", "ck_type"}
SMELL_TYPE = {"blob": "class", "data class": "class",
              "feature envy": "function", "long method": "function"}

feat = pd.read_csv(PROCESSED / "features_full_repaired.csv", low_memory=False)
lab = pd.read_csv(ROOT / "data" / "interim" / "labeled_rows_with_files.csv")[
    ["id", "repository", "sample_id"]]
use = feat.merge(lab, on="id", how="left")
use = use[use["usable"]].copy()


def numeric_features(sub):
    cols, dropped = {}, {}
    for c in [c for c in sub.columns if c.startswith("ck_") and c not in NON_FEATURES]:
        s = sub[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.replace({"True": 1, "False": 0, True: 1, False: 0})
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().mean() < 0.99:
            dropped[c] = "mostly missing / non-numeric"
        elif s.nunique(dropna=True) <= 1:
            dropped[c] = "constant"
        else:
            cols[c] = s
    return pd.DataFrame(cols, index=sub.index), dropped


# ---- features per CK level (class.csv metrics for class smells, method.csv for function smells)
features_by_type, dropped_by_type, parts = {}, {}, []
META = ["id", "smell", "sample_id", "repository", "file_id", "code_name", "type", "label"]
for t in ("class", "function"):
    sub = use[use["type"] == t]
    X, dropped = numeric_features(sub)
    features_by_type[t] = list(X.columns)
    dropped_by_type[t] = dropped
    parts.append(pd.concat([sub[META], X], axis=1))

# ---- aggregate to one row per code location, majority-vote label
locs, agg_report = [], []
for part in parts:
    feats = [c for c in part.columns if c.startswith("ck_")]
    agg = {**{c: "first" for c in ["repository", "file_id", "code_name", "type"] + feats},
           "label": "mean", "id": "count"}
    g = part.groupby(["smell", "sample_id"], sort=False).agg(agg).reset_index()
    g = g.rename(columns={"label": "label_mean", "id": "n_reviews"})
    locs.append(g)
all_locs = pd.concat(locs, ignore_index=True)
tie = all_locs["label_mean"] == 0.5
for s, sub in all_locs.groupby("smell"):
    kept = sub[sub["label_mean"] != 0.5]
    agg_report.append({"smell": s, "usable_reviews": int((use["smell"] == s).sum()),
                       "locations": len(sub), "ties_dropped": int((sub["label_mean"] == 0.5).sum()),
                       "final_rows": len(kept), "positives": int((kept["label_mean"] > 0.5).sum())})
all_locs = all_locs[~tie].copy().reset_index(drop=True)
all_locs["label"] = (all_locs["label_mean"] > 0.5).astype(int)
rep = pd.DataFrame(agg_report)
rep["pos_rate"] = (rep["positives"] / rep["final_rows"]).round(3)
print("=== Aggregation: reviews -> one row per code location ===")
print(rep.to_string(index=False))

# ---- repository-aware split: one repo -> exactly one split / CV fold, for all smells
n = len(all_locs)
strat = (all_locs["smell"] + "_" + all_locs["label"].astype(str)).to_numpy()
groups = all_locs["repository"].to_numpy()
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, test_idx = next(sgkf.split(np.zeros(n), strat, groups))
split = np.array(["train"] * n, dtype=object)
split[test_idx] = "test"
cv_fold = np.full(n, -1)
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, val) in enumerate(sgkf2.split(np.zeros(len(train_idx)), strat[train_idx], groups[train_idx])):
    cv_fold[train_idx[val]] = k
all_locs["split"], all_locs["cv_fold"] = split, cv_fold

chk = all_locs.groupby("repository").agg(n_split=("split", "nunique"), n_cv=("cv_fold", "nunique"))
assert (chk["n_split"] == 1).all() and (chk["n_cv"] == 1).all(), "a repository leaked across splits"
all_locs.groupby("repository")[["split", "cv_fold"]].first().reset_index().to_csv(
    PROCESSED / "repo_split.csv", index=False)
print(f"\nrepositories: {all_locs['repository'].nunique()} | train repos: "
      f"{all_locs[all_locs.split == 'train']['repository'].nunique()} | test repos: "
      f"{all_locs[all_locs.split == 'test']['repository'].nunique()} | repo-leakage check: PASSED")

# ---- write per-smell datasets
for s, t in SMELL_TYPE.items():
    sub = all_locs[all_locs["smell"] == s]
    feats = features_by_type[t]
    cols = ["sample_id", "repository", "file_id", "code_name", "label", "label_mean",
            "n_reviews", "split", "cv_fold"] + feats
    out = sub[cols].rename(columns={f: f[3:] for f in feats})
    out.to_csv(OUT / f"{s.replace(' ', '_')}.csv", index=False)

(PROCESSED / "feature_lists.json").write_text(json.dumps({
    "features": {t: [f[3:] for f in fs] for t, fs in features_by_type.items()},
    "dropped": dropped_by_type, "smell_to_type": SMELL_TYPE}, indent=2))

# ---- verification output
print("\n=== Split sizes: rows (positives) per smell ===")
tab = all_locs.pivot_table(index="smell", columns="split", values="label",
                           aggfunc=["size", "sum"]).fillna(0).astype(int)
print(tab.to_string())
print("\n=== Positives per CV fold inside train ===")
trn = all_locs[all_locs["split"] == "train"]
print(trn.pivot_table(index="smell", columns="cv_fold", values="label", aggfunc="sum").astype(int).to_string())
print("\n=== Rows per CV fold inside train ===")
print(trn.pivot_table(index="smell", columns="cv_fold", values="label", aggfunc="size").to_string())

print("\n=== Features ===")
for t in ("class", "function"):
    print(f"{t}: {len(features_by_type[t])} features")
    print("  ", [f[3:] for f in features_by_type[t]])
    print("   dropped:", {k[3:]: v for k, v in dropped_by_type[t].items()})

print("\n=== Leakage / quality checks per smell ===")
for s, t in SMELL_TYPE.items():
    sub = all_locs[all_locs["smell"] == s]
    feats = features_by_type[t]
    h = pd.util.hash_pandas_object(sub[feats].round(6).fillna(-1), index=False)
    train_h = set(h[sub["split"] == "train"])
    test_in_train = int(h[sub["split"] == "test"].isin(train_h).sum())
    nan_max = sub[feats].isna().mean().max()
    print(f"{s}: test rows whose exact feature vector also appears in train: {test_in_train} "
          f"of {int((sub['split'] == 'test').sum())} | max NaN fraction in any feature: {nan_max:.4f}")

print(f"\nSaved datasets to {OUT} | split table: {PROCESSED / 'repo_split.csv'}")