"""
Stage 4 / Step 2: baselines + first models, repo-grouped 5-fold CV on the train split.
Model selection uses CV only (average precision). The locked test split is evaluated
once: for the two baselines and the CV-selected model. Threshold comes from train OOF.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "dataset"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
SEED = 42
SMELLS = ["blob", "data_class", "feature_envy", "long_method"]
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
        "hist_gb": Pipeline([("imp", imp()),
                             ("clf", HistGradientBoostingClassifier(
                                 learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                 l2_regularization=1.0, class_weight="balanced",
                                 random_state=SEED))]),
    }


BASELINES = {
    "dummy": DummyClassifier(strategy="prior"),
    "stump": DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=SEED),
}


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def at_threshold(y, p, t):
    yhat = (p >= t).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, float(matthews_corrcoef(y, yhat))


def oof_proba(model, X, y, folds):
    p, fold_ap = np.zeros(len(y)), []
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        m = clone(model).fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
        fold_ap.append(average_precision_score(y[va], p[va]))
    return p, np.array(fold_ap)


def boot_ap(y, p, n=1000):
    rng, out = np.random.default_rng(SEED), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        out.append(average_precision_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


def dup_mask(Xtr, Xte):
    h_tr = set(pd.util.hash_pandas_object(pd.DataFrame(Xtr).round(6), index=False))
    h_te = pd.util.hash_pandas_object(pd.DataFrame(Xte).round(6), index=False)
    return h_te.isin(h_tr).to_numpy()


cv_rows, test_rows = [], []
for smell in SMELLS:
    df = pd.read_csv(DATA / f"{smell}.csv")
    feats = [c for c in df.columns if c not in META]
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    Xtr, ytr, folds = tr[feats].to_numpy(float), tr["label"].to_numpy(int), tr["cv_fold"].to_numpy(int)
    Xte, yte = te[feats].to_numpy(float), te["label"].to_numpy(int)
    assert (Xtr >= 0).all() and (Xte >= 0).all(), "negative feature values: log1p unsafe"

    print("=" * 90)
    print(f"{smell}: train {len(ytr)} rows ({ytr.sum()} pos, {ytr.mean():.3f}) | "
          f"test {len(yte)} rows ({yte.sum()} pos, {yte.mean():.3f}) | {len(feats)} features")

    all_models = {**BASELINES, **make_models()}
    res = {}
    for name, model in all_models.items():
        p, fap = oof_proba(model, Xtr, ytr, folds)
        thr = 0.0 if name == "dummy" else best_threshold(ytr, p)   # dummy = flag everything
        prec, rec, f1, mcc = at_threshold(ytr, p, thr)
        res[name] = dict(p=p, thr=thr, ap=average_precision_score(ytr, p),
                         roc=roc_auc_score(ytr, p), fm=fap.mean(), fs=fap.std(),
                         prec=prec, rec=rec, f1=f1, mcc=mcc)
        cv_rows.append({"smell": smell, "model": name, "oof_ap": res[name]["ap"],
                        "fold_ap_mean": fap.mean(), "fold_ap_std": fap.std(),
                        "oof_roc_auc": res[name]["roc"], "threshold": thr,
                        "oof_precision": prec, "oof_recall": rec, "oof_f1": f1, "oof_mcc": mcc})

    print("\nCross-validation on train (out-of-fold, repo-grouped):")
    tbl = pd.DataFrame([{"model": n, "AP": r["ap"], "fold AP": f'{r["fm"]:.3f}+/-{r["fs"]:.3f}',
                         "ROC-AUC": r["roc"], "thr": r["thr"], "P": r["prec"], "R": r["rec"],
                         "F1": r["f1"], "MCC": r["mcc"]} for n, r in res.items()]).round(3)
    print(tbl.to_string(index=False))

    best = max(make_models().keys(), key=lambda n: res[n]["ap"])
    print(f"\nSelected by CV average precision: {best}")

    # stump rule, for explanation
    stump = clone(BASELINES["stump"]).fit(Xtr, ytr)
    f, t = stump.tree_.feature[0], stump.tree_.threshold[0]
    left, right = Xtr[:, f] <= t, Xtr[:, f] > t
    print(f"Stump rule: {feats[f]} <= {t:.1f}  -> smell rate {ytr[left].mean():.3f} (n={left.sum()}) | "
          f"{feats[f]} > {t:.1f} -> smell rate {ytr[right].mean():.3f} (n={right.sum()})")

    print("\nTest set (evaluated once):")
    final_model = None
    for name in ["dummy", "stump", best]:
        m = clone(all_models[name]).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        prec, rec, f1, mcc = at_threshold(yte, p, res[name]["thr"])
        lo, hi = boot_ap(yte, p)
        ap = average_precision_score(yte, p)
        test_rows.append({"smell": smell, "model": name, "test_ap": ap, "ap_ci_low": lo,
                          "ap_ci_high": hi, "test_roc_auc": roc_auc_score(yte, p),
                          "precision": prec, "recall": rec, "f1": f1, "mcc": mcc,
                          "n_test": len(yte), "n_pos": int(yte.sum())})
        print(f"  {name:14s} AP {ap:.3f} (95% CI {lo:.3f}-{hi:.3f}) | ROC-AUC "
              f"{roc_auc_score(yte, p):.3f} | P {prec:.3f} R {rec:.3f} F1 {f1:.3f} MCC {mcc:.3f}")
        if name == best:
            final_model, p_best = m, p

    dm = dup_mask(Xtr, Xte)
    keep = ~dm
    if dm.sum() and yte[keep].sum() > 0:
        print(f"  sensitivity: {int(dm.sum())} test rows share an exact feature vector with train "
              f"({int(yte[dm].sum())} positive); AP without them: "
              f"{average_precision_score(yte[keep], p_best[keep]):.3f}")

    joblib.dump({"model": final_model, "features": feats, "threshold": res[best]["thr"],
                 "smell": smell, "model_name": best}, MODELS / f"{smell}.joblib")

pd.DataFrame(cv_rows).to_csv(REPORTS / "stage4_cv_results.csv", index=False)
pd.DataFrame(test_rows).to_csv(REPORTS / "stage4_test_results.csv", index=False)
print("\n" + "=" * 90)
print("Saved models to", MODELS, "and results to", REPORTS)