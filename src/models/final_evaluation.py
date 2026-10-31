"""
Stage 6: final evaluation artifacts for the four saved models. Frozen models and thresholds;
nothing is tuned or selected here. Asserts that test AP equals the first-run result.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "dataset"
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SMELLS = ["blob", "data_class", "feature_envy", "long_method"]
TITLES = {"blob": "Blob (God Class)", "data_class": "Data Class",
          "feature_envy": "Feature Envy", "long_method": "Long Method"}
SEED = 42
saved = pd.read_csv(REPORTS / "stage4_test_results.csv")


def boot_ap(y, p, n=1000):
    rng, out = np.random.default_rng(SEED), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            out.append(average_precision_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


summary, importances, curves, cms = [], {}, {}, {}
for smell in SMELLS:
    art = joblib.load(MODELS / f"{smell}.joblib")
    model, feats, thr, name = art["model"], art["features"], art["threshold"], art["model_name"]
    te = pd.read_csv(DATA / f"{smell}.csv")
    te = te[te["split"] == "test"]
    X, y = te[feats].to_numpy(float), te["label"].to_numpy(int)
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= thr).astype(int)

    ap = average_precision_score(y, p)
    ref = float(saved[(saved["smell"] == smell) & (saved["model"] == name)]["test_ap"].iloc[0])
    assert abs(ap - ref) < 5e-4, f"{smell}: test AP {ap:.4f} differs from first run {ref:.4f}"

    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    lo, hi = boot_ap(y, p)
    summary.append({"smell": TITLES[smell], "model": name, "n_test": len(y), "n_positive": int(y.sum()),
                    "threshold": round(thr, 3), "AP": round(ap, 3), "AP_ci_low": round(lo, 3),
                    "AP_ci_high": round(hi, 3), "ROC_AUC": round(roc_auc_score(y, p), 3),
                    "precision": round(prec, 3), "recall": round(rec, 3),
                    "F1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                    "MCC": round(float(matthews_corrcoef(y, yhat)), 3),
                    "specificity": round(tn / (tn + fp), 3), "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    cms[smell] = np.array([[tn, fp], [fn, tp]])
    pr, rc, _ = precision_recall_curve(y, p)
    curves[smell] = (rc, pr, y.mean(), rec, prec)

    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp, kind = pd.Series(clf.feature_importances_, index=feats), "importance"
    else:
        imp, kind = pd.Series(clf.coef_[0], index=feats), "coefficient"
    importances[smell] = (imp.reindex(imp.abs().sort_values(ascending=False).index), kind)

res = pd.DataFrame(summary)
res.to_csv(REPORTS / "final_evaluation_summary.csv", index=False)
print("=== Final test-set results (frozen models; first-run AP reproduced) ===")
print(res.drop(columns=["TP", "FP", "FN", "TN"]).to_string(index=False))
print("\n=== Confusion counts at each model's threshold ===")
print(res[["smell", "TP", "FP", "FN", "TN"]].to_string(index=False))

# ---- figures
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, smell in zip(axes.ravel(), SMELLS):
    rc, pr, base, r_at, p_at = curves[smell]
    ax.plot(rc, pr, lw=1.8)
    ax.axhline(base, ls="--", c="gray", lw=1, label=f"random ({base:.2f})")
    ax.scatter([r_at], [p_at], c="red", zorder=3, label="chosen threshold")
    ax.set(title=TITLES[smell], xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "pr_curves_test.png", dpi=150)

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, smell in zip(axes.ravel(), SMELLS):
    cm = cms[smell]
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(title=TITLES[smell], xticks=[0, 1], yticks=[0, 1],
           xticklabels=["no smell", "smell"], yticklabels=["no smell", "smell"],
           xlabel="predicted", ylabel="actual")
fig.tight_layout()
fig.savefig(FIG / "confusion_matrices_test.png", dpi=150)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
print("\n=== Top features per model ===")
rows = []
for ax, smell in zip(axes.ravel(), SMELLS):
    imp, kind = importances[smell]
    top = imp.head(8)
    print(f"\n{TITLES[smell]} ({kind}):")
    print(top.round(3).to_string())
    rows += [{"smell": TITLES[smell], "feature": f, "value": v, "kind": kind} for f, v in imp.items()]
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title(f"{TITLES[smell]} - top 8 ({kind})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "top_features.png", dpi=150)
pd.DataFrame(rows).to_csv(REPORTS / "feature_importance.csv", index=False)
print(f"\nSaved figures to {FIG} and tables to {REPORTS}")