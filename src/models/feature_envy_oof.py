"""
Feature Envy: out-of-fold (OOF) predictions on the TRAIN split only.

Uses the existing cv_fold column and the same model definitions and seed as
train_baselines.py. Does not touch the test split, saved models, or reports.
A reproduction check against reports/stage4_cv_results.csv must pass before
anything is written.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "dataset"
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "data" / "processed" / "oof"
SEED = 42
TOL = 5e-4
META = ["sample_id", "repository", "file_id", "code_name", "label", "label_mean",
        "n_reviews", "split", "cv_fold"]


def make_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "logreg": Pipeline([("imp", imp()), ("log", FunctionTransformer(np.log1p)),
                            ("sc", StandardScaler()),
                            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
        "random_forest": Pipeline([("imp", imp()),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=300, min_samples_leaf=3,
                                       class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=SEED))]),
        "stump": DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=SEED),
    }


def oof_proba(model, X, y, folds):
    p, fold_ap = np.zeros(len(y)), []
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        m = clone(model).fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
        fold_ap.append(average_precision_score(y[va], p[va]))
    return p, np.array(fold_ap)


df = pd.read_csv(DATA / "feature_envy.csv")
feats = [c for c in df.columns if c not in META]
tr = df[df["split"] == "train"].reset_index(drop=True)
X = tr[feats].to_numpy(float)
y = tr["label"].to_numpy(int)
folds = tr["cv_fold"].to_numpy(int)
assert (X >= 0).all(), "negative feature values: log1p unsafe"
print(f"train rows: {len(tr)} | positives: {int(y.sum())} | features: {len(feats)} | "
      f"folds: {sorted(set(folds))}")

# ---- compute OOF predictions
probs, now = {}, {}
for name, model in make_models().items():
    p, fap = oof_proba(model, X, y, folds)
    probs[name] = p
    now[name] = {"oof_ap": average_precision_score(y, p),
                 "fold_ap_mean": fap.mean(), "fold_ap_std": fap.std()}

# ---- reproduction check against the saved CV results
saved = pd.read_csv(REPORTS / "stage4_cv_results.csv")
saved = saved[saved["smell"] == "feature_envy"].set_index("model")
print("\n=== Reproduction check vs reports/stage4_cv_results.csv ===")
print(f"{'model':14s} {'metric':13s} {'now':>8s} {'saved':>8s} {'diff':>9s}  status")
ok = True
for name in now:
    for metric in ("oof_ap", "fold_ap_mean", "fold_ap_std"):
        a, b = now[name][metric], float(saved.loc[name, metric])
        good = abs(a - b) <= TOL
        ok &= good
        print(f"{name:14s} {metric:13s} {a:8.4f} {b:8.4f} {a - b:9.5f}  {'OK' if good else 'MISMATCH'}")

if not ok:
    print("\nSTOP: out-of-fold results do not reproduce the saved CV results. Nothing was written.")
    print("Most likely causes: (1) data/processed/dataset/feature_envy.csv or the cv_fold "
          "assignment changed since the first run (e.g. build_datasets.py rerun);")
    print("(2) scikit-learn/numpy versions changed; (3) train_baselines.py model settings or "
          "seed were edited. Tell me which of these applies before we continue.")
    sys.exit(1)
print("\nREPRODUCTION CHECK PASSED")

# ---- attach MLCQ line ranges (for reading source later) and write the OOF table
lab = pd.read_csv(ROOT / "data" / "interim" / "labeled_rows_with_files.csv",
                  usecols=["sample_id", "smell", "start_line", "end_line"])
lab = lab[lab["smell"] == "feature envy"]
n_conflict = int(lab.drop_duplicates(["sample_id", "start_line", "end_line"])
                 ["sample_id"].duplicated().sum())
lines = lab.drop_duplicates("sample_id")[["sample_id", "start_line", "end_line"]]
out = tr[["sample_id", "repository", "file_id", "code_name", "label", "label_mean",
          "n_reviews", "cv_fold"]].merge(lines, on="sample_id", how="left", validate="many_to_one")
assert out["start_line"].notna().all(), "some sample_ids have no line range"
for name, p in probs.items():
    out[f"p_{name}"] = p
out = pd.concat([out, tr[feats]], axis=1)
assert len(out) == len(tr) and out[[f"p_{n}" for n in probs]].notna().all().all()

OUT_DIR.mkdir(parents=True, exist_ok=True)
path = OUT_DIR / "feature_envy_train_oof.csv"
out.to_csv(path, index=False)

print(f"\nsample_ids with more than one line range in MLCQ: {n_conflict}")
print(f"rows written: {len(out)} ({int(out['label'].sum())} positives) -> {path}")
print("positives per CV fold:", out.groupby("cv_fold")["label"].sum().to_dict())
print("\nfirst 3 rows:")
print(out[["sample_id", "label", "n_reviews", "cv_fold", "start_line", "end_line",
           "p_logreg", "p_random_forest", "p_stump"]].head(3).round(3).to_string(index=False))