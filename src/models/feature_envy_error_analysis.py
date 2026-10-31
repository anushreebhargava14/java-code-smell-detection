"""
Feature Envy error analysis on TRAIN out-of-fold predictions only. Read-only:
no test data, no models, no reports touched. Writes two small CSVs next to the OOF file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parents[2]
OOF_DIR = ROOT / "data" / "processed" / "oof"
SRC = ROOT / "data" / "raw" / "source_files"

oof = pd.read_csv(OOF_DIR / "feature_envy_train_oof.csv")
cv = pd.read_csv(ROOT / "reports" / "stage4_cv_results.csv")
THR = float(cv[(cv["smell"] == "feature_envy") & (cv["model"] == "logreg")]["threshold"].iloc[0])

NON_FEATURES = {"sample_id", "repository", "file_id", "code_name", "label", "label_mean",
                "n_reviews", "cv_fold", "start_line", "end_line",
                "p_logreg", "p_random_forest", "p_stump"}
feats = [c for c in oof.columns if c not in NON_FEATURES]
y = oof["label"].to_numpy(int)
n = len(oof)
pos, neg = oof[oof["label"] == 1], oof[oof["label"] == 0]
PROFILE = [c for c in ["loc", "rfc", "fanout", "cbo", "wmc", "parametersQty",
                       "methodsInvokedQty", "methodsInvokedLocalQty"] if c in oof.columns]


def tail(s, w=58):
    s = str(s)
    return s if len(s) <= w else "..." + s[-(w - 3):]


def show_source(r, max_lines=10, width=100):
    p = SRC / f"{int(r['file_id'])}.java"
    if not p.exists():
        print("    (source file missing)")
        return
    lines = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").split("\n")
    seg = lines[int(r["start_line"]) - 1: int(r["end_line"])]
    for ln in seg[:max_lines]:
        print("    |", ln[:width].rstrip())
    if len(seg) > max_lines:
        print(f"    | ... ({len(seg) - max_lines} more lines)")


def describe(r):
    return (f"sample_id={int(r['sample_id'])} {tail(r['code_name'])} | p_logreg={r['p_logreg']:.3f} "
            f"p_rf={r['p_random_forest']:.3f} | label_mean={r['label_mean']:.2f} "
            f"n_reviews={int(r['n_reviews'])} | lines {int(r['start_line'])}-{int(r['end_line'])} "
            f"| loc={r['loc']:.0f} rfc={r['rfc']:.0f} fanout={r['fanout']:.0f}")


print(f"train rows {n} | positives {int(y.sum())} ({y.mean():.3f}) | logreg threshold from CV: {THR:.3f}")

# ---------------------------------------------------------------- 1. ranking usefulness
print("\n=== 1. If a reviewer inspects only the top-scored methods (OOF scores) ===")
rows = []
for model in ("p_logreg", "p_random_forest"):
    rank = oof[model].rank(ascending=False, method="first")
    for label, k in [("top 1%", round(0.01 * n)), ("top 5%", round(0.05 * n)),
                     ("top 10%", round(0.10 * n)), ("top 20", 20), ("top 50", 50)]:
        found = int(y[(rank <= k).to_numpy()].sum())
        rows.append({"model": model, "budget": label, "flagged": k, "positives_found": found,
                     "recall": found / y.sum(), "precision": found / k,
                     "lift_vs_random": (found / k) / y.mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- 2/3. outcome groups
flag = oof["p_logreg"] >= THR
groups = {"TP (positive, flagged)": pos[(pos["p_logreg"] >= THR)],
          "FN (positive, missed)": pos[(pos["p_logreg"] < THR)],
          "FP (negative, flagged)": neg[(neg["p_logreg"] >= THR)],
          "TN (negative, not flagged)": neg[(neg["p_logreg"] < THR)]}
print(f"\n=== 2. Outcomes at the CV threshold ({THR:.3f}) ===")
prof = pd.DataFrame({name: {"n": len(g), **{f"median {c}": g[c].median() for c in PROFILE}}
                     for name, g in groups.items()}).T
print(prof.round(1).to_string())

print("\n=== 3a. Lowest-scoring POSITIVES (what the model misses most) ===")
fn = pos.sort_values("p_logreg").head(12)
fn.to_csv(OOF_DIR / "feature_envy_false_negatives.csv", index=False)
for _, r in fn.iterrows():
    print(" -", describe(r))
print("\nsource of the 5 lowest-scoring positives:")
for _, r in fn.head(5).iterrows():
    print(f"  [{tail(r['code_name'], 70)}]")
    show_source(r)

print("\n=== 3b. Highest-scoring NEGATIVES (confident false alarms) ===")
fp = neg.sort_values("p_logreg", ascending=False).head(25)
fp.to_csv(OOF_DIR / "feature_envy_false_positives.csv", index=False)
for _, r in fp.head(12).iterrows():
    print(" -", describe(r))
print("\nsource of the 5 highest-scoring negatives:")
for _, r in fp.head(5).iterrows():
    print(f"  [{tail(r['code_name'], 70)}]")
    show_source(r)

print("\n=== 3c. How many 'false alarms' had at least one reviewer vote 'smell'? ===")
multi_neg = neg[neg["n_reviews"] >= 2]
top50 = neg.sort_values("p_logreg", ascending=False).head(50)
top50_multi = top50[top50["n_reviews"] >= 2]
print(f"all negatives with >=2 reviews: {len(multi_neg)} | share with a dissenting 'smell' vote: "
      f"{(multi_neg['label_mean'] > 0).mean():.3f}")
print(f"top-50 scored negatives: {len(top50_multi)} have >=2 reviews | share with a dissenting "
      f"'smell' vote: {(top50_multi['label_mean'] > 0).mean():.3f}")
print(f"top-50 scored negatives that are single-review: {int((top50['n_reviews'] == 1).sum())} of 50")

# ---------------------------------------------------------------- 4. label clarity
print("\n=== 4. Does the model find the clear positives? (AP against all negatives) ===")
subsets = {"all positives": pos,
           "unanimous (label_mean = 1)": pos[pos["label_mean"] == 1.0],
           "majority-only (label_mean < 1)": pos[pos["label_mean"] < 1.0],
           "single-review positives": pos[pos["n_reviews"] == 1],
           "multi-review positives": pos[pos["n_reviews"] > 1]}
rows = []
for name, g in subsets.items():
    r = {"positive group": name, "n": len(g)}
    for model in ("p_logreg", "p_random_forest"):
        yy = np.r_[np.ones(len(g)), np.zeros(len(neg))]
        pp = np.r_[g[model], neg[model]]
        r[f"AP {model[2:]}"] = average_precision_score(yy, pp)
    r["recall @ threshold"] = float((g["p_logreg"] >= THR).mean())
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("(random-guess AP for a subset ~ its n / (n + negatives))")

# ---------------------------------------------------------------- 5. univariate signal
X = oof[feats].copy()
if {"methodsInvokedLocalQty", "methodsInvokedQty"} <= set(X.columns):
    tot, loc_calls = X["methodsInvokedQty"], X["methodsInvokedLocalQty"]
    X["DERIVED_foreign_calls"] = tot - loc_calls
    X["DERIVED_local_call_ratio"] = np.where(tot > 0, loc_calls / tot.clip(lower=1), 0.0)
if {"fanout", "loc"} <= set(X.columns):
    X["DERIVED_fanout_per_loc"] = X["fanout"] / X["loc"].clip(lower=1)
rows = []
for c in X.columns:
    x = X[c].to_numpy(float)
    auc = roc_auc_score(y, x)
    sign = 1 if auc >= 0.5 else -1
    rows.append({"feature": c, "AUC": auc, "strength": abs(auc - 0.5) + 0.5,
                 "direction": "higher => smell" if sign == 1 else "lower => smell",
                 "AP (single feature)": average_precision_score(y, sign * x)})
uni = pd.DataFrame(rows).sort_values("strength", ascending=False).drop(columns="strength")
print("\n=== 5. Single-feature signal on train (AUC, AP; base rate AP = %.3f) ===" % y.mean())
print(uni.round(3).to_string(index=False))

# ---------------------------------------------------------------- 6. concentration
print("\n=== 6. Where do the positives come from? ===")
per_repo = pos.groupby("repository").size().sort_values(ascending=False)
print(f"positives: {len(pos)} from {len(per_repo)} repositories | top 5 repos hold "
      f"{per_repo.head(5).sum()} positives ({per_repo.head(5).sum() / len(pos):.2f})")
print(per_repo.head(5).to_string())
print("positives by number of reviews:", pos["n_reviews"].value_counts().sort_index().to_dict())
print("positives by label_mean:", pos["label_mean"].round(2).value_counts().sort_index().to_dict())
print(f"\nCSVs written: {OOF_DIR / 'feature_envy_false_negatives.csv'}, "
      f"{OOF_DIR / 'feature_envy_false_positives.csv'}")