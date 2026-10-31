"""
Feature Envy limited tuning. TRAIN split only. Repeated repo-grouped CV (5 folds x 5 repeats),
feature selection INSIDE each training fold. The test split is never loaded; no models saved.
Repeat 0 uses the existing cv_fold, so the current models must reproduce the saved CV results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SEED, REPEATS, TOL = 42, 5, 5e-4
META = ["sample_id", "repository", "file_id", "code_name", "label", "label_mean",
        "n_reviews", "split", "cv_fold"]

df = pd.read_csv(ROOT / "data" / "processed" / "dataset" / "feature_envy.csv")
feats = [c for c in df.columns if c not in META]
tr = df[df["split"] == "train"].reset_index(drop=True)          # test rows are never used
X, y = tr[feats].to_numpy(float), tr["label"].to_numpy(int)
groups = tr["repository"].to_numpy()
assert (X >= 0).all()
n, n_pos = len(y), int(y.sum())

# ---- fold sets: repeat 0 = existing cv_fold; repeats 1..4 = new repo-grouped stratified splits
fold_sets = [tr["cv_fold"].to_numpy(int)]
for r in range(1, REPEATS):
    f = np.full(n, -1)
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
    for k, (_, va) in enumerate(sg.split(np.zeros(n), y, groups)):
        f[va] = k
    fold_sets.append(f)
print(f"train rows {n} | positives {n_pos} | features {len(feats)} | repeats {REPEATS}")
print("min positives in any validation fold, per repeat:",
      [min(int(y[f == k].sum()) for k in range(5)) for f in fold_sets])


def logreg(k, C):
    steps = [("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p)),
             ("sc", StandardScaler())]
    if k != "all":
        steps.append(("sel", SelectKBest(f_classif, k=k)))
    steps.append(("clf", LogisticRegression(class_weight="balanced", C=C, max_iter=3000)))
    return Pipeline(steps)


def forest(leaf):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf,
                                                    class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=SEED))])


configs = []
for k in [1, 2, 3, 5, 8, "all"]:
    for C in [0.03, 0.1, 0.3, 1.0]:
        configs.append({"id": f"logreg_k={k}_C={C}", "family": "logreg", "k": k, "reg": C,
                        "model": logreg(k, C)})
for leaf in [3, 5, 15, 30]:
    configs.append({"id": f"rf_leaf={leaf}", "family": "rf", "k": "-", "reg": leaf,
                    "model": forest(leaf)})


def evaluate(model):
    pooled_ap, pooled_roc, rec5, prec5, fold_ap = [], [], [], [], []
    budget = round(0.05 * n)
    for folds in fold_sets:
        p = np.zeros(n)
        for k in range(5):
            trn, va = folds != k, folds == k
            m = clone(model).fit(X[trn], y[trn])
            p[va] = m.predict_proba(X[va])[:, 1]
            fold_ap.append(average_precision_score(y[va], p[va]))
        top = np.argsort(-p, kind="stable")[:budget]
        pooled_ap.append(average_precision_score(y, p))
        pooled_roc.append(roc_auc_score(y, p))
        rec5.append(y[top].sum() / n_pos)
        prec5.append(y[top].sum() / budget)
    fa = np.array(fold_ap)
    return {"pooled_ap_r0": pooled_ap[0], "pooled_ap_mean": np.mean(pooled_ap),
            "pooled_ap_sd": np.std(pooled_ap), "fold_ap_mean": fa.mean(),
            "fold_ap_se": fa.std(ddof=1) / np.sqrt(len(fa)), "roc_auc": np.mean(pooled_roc),
            "recall_top5": np.mean(rec5), "precision_top5": np.mean(prec5)}


# ---- reproduction check on the two current models (repeat 0 == the original protocol)
saved = pd.read_csv(ROOT / "reports" / "stage4_cv_results.csv")
saved = saved[saved["smell"] == "feature_envy"].set_index("model")["oof_ap"]
results = {}
ok = True
for cid, name in [("logreg_k=all_C=1.0", "logreg"), ("rf_leaf=3", "random_forest")]:
    cfg = next(c for c in configs if c["id"] == cid)
    results[cid] = evaluate(cfg["model"])
    a, b = results[cid]["pooled_ap_r0"], float(saved[name])
    print(f"reproduction {name}: repeat-0 pooled AP {a:.4f} vs saved {b:.4f} "
          f"-> {'OK' if abs(a - b) <= TOL else 'MISMATCH'}")
    ok &= abs(a - b) <= TOL
if not ok:
    print("STOP: current models do not reproduce the saved CV results; tuning not run.")
    sys.exit(1)

for cfg in configs:
    if cfg["id"] not in results:
        results[cfg["id"]] = evaluate(cfg["model"])

res = pd.DataFrame([{"id": c["id"], "family": c["family"], "k": c["k"], "reg": c["reg"],
                     **results[c["id"]]} for c in configs])
res.to_csv(ROOT / "reports" / "feature_envy_tuning_cv.csv", index=False)

show = res.sort_values("fold_ap_mean", ascending=False).copy()
show["fold AP (mean+/-SE)"] = show.apply(lambda r: f"{r.fold_ap_mean:.3f}+/-{r.fold_ap_se:.3f}", axis=1)
show["pooled AP (mean+/-sd)"] = show.apply(lambda r: f"{r.pooled_ap_mean:.3f}+/-{r.pooled_ap_sd:.3f}", axis=1)
cols = ["id", "fold AP (mean+/-SE)", "pooled AP (mean+/-sd)", "roc_auc", "recall_top5", "precision_top5"]
print("\n=== Top 12 configs by mean per-fold AP (5 repeats x 5 folds) ===")
print(show[cols].head(12).round(3).to_string(index=False))
print("\n=== Current models, for reference ===")
print(show[show["id"].isin(["logreg_k=all_C=1.0", "rf_leaf=3"])][cols].round(3).to_string(index=False))
print("\n=== Logistic regression: mean per-fold AP by k (rows) and C (columns) ===")
print(res[res["family"] == "logreg"].pivot(index="k", columns="reg", values="fold_ap_mean")
      .reindex([1, 2, 3, 5, 8, "all"]).round(3).to_string())
print("\n=== Random forest: mean per-fold AP by min_samples_leaf ===")
print(res[res["family"] == "rf"][["reg", "fold_ap_mean", "fold_ap_se"]].round(3).to_string(index=False))

# ---- pre-declared selection rule
best = res.sort_values("fold_ap_mean", ascending=False).iloc[0]
eligible = res[res["fold_ap_mean"] >= best["fold_ap_mean"] - best["fold_ap_se"]]


def complexity(r):                       # lower = simpler
    if r["family"] == "logreg":
        return (0, 999 if r["k"] == "all" else int(r["k"]), float(r["reg"]))
    return (1, 0, -float(r["reg"]))


pick = min((r for _, r in eligible.iterrows()), key=complexity)
base = res[res["id"] == "logreg_k=all_C=1.0"].iloc[0]
delta = pick["fold_ap_mean"] - base["fold_ap_mean"]
print("\n=== Pre-declared selection rule ===")
print(f"best mean per-fold AP: {best['id']} ({best['fold_ap_mean']:.3f}, SE {best['fold_ap_se']:.3f}); "
      f"{len(eligible)} configs within 1 SE")
print(f"simplest eligible config: {pick['id']} ({pick['fold_ap_mean']:.3f})")
print(f"current model (logreg_k=all_C=1.0): {base['fold_ap_mean']:.3f} (SE {base['fold_ap_se']:.3f})")
print(f"improvement of pick over current: {delta:+.3f} -> "
      f"{'SWITCH (exceeds current model SE)' if (pick['id'] != base['id'] and delta > base['fold_ap_se']) else 'KEEP CURRENT'}")
print("\nSaved reports/feature_envy_tuning_cv.csv (test split never loaded; no models saved)")