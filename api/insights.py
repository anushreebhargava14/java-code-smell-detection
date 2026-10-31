"""
Read-only data layer for the web UI.

Everything returned here is computed from files the pipeline already produced
(reports/, data/processed/, data/raw/, data/interim/, models/). Nothing is hard-coded
and no model is retrained. Results are cached after the first call.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "inference"))
import predict as P  # noqa: E402

REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
DATASET = PROCESSED / "dataset"
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
SMELLS = list(P.SMELL_LEVEL)
TITLE_TO_KEY = {v: k for k, v in P.TITLES.items()}
MAX_EXAMPLE_BYTES = 250_000


def _records(df):
    """DataFrame -> list of plain dicts with NaN converted to null (valid JSON)."""
    return json.loads(df.to_json(orient="records"))


def _read(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


def _dataset_csv(smell):
    return pd.read_csv(DATASET / f"{smell}.csv")


def short_name(code_name, level):
    head = str(code_name).strip().split(" ")[0]
    if level == "class":
        return head.replace("$", ".").split(".")[-1]
    if "#" in head:
        cls, m = head.rsplit("#", 1)
    elif "." in head:
        cls, m = head.rsplit(".", 1)
    else:
        return head
    return f"{cls.replace('$', '.').split('.')[-1]}.{m}"


@lru_cache(maxsize=1)
def results():
    out = {"summary": [], "cv": [], "test": [], "importance": {}}
    summary = _read(REPORTS / "final_evaluation_summary.csv")
    if summary is not None:
        summary = summary.copy()
        summary["key"] = summary["smell"].map(TITLE_TO_KEY)
        out["summary"] = _records(summary)
    cv = _read(REPORTS / "stage4_cv_results.csv")
    if cv is not None:
        out["cv"] = _records(cv)
    test = _read(REPORTS / "stage4_test_results.csv")
    if test is not None:
        out["test"] = _records(test)
    imp = _read(REPORTS / "feature_importance.csv")
    if imp is not None:
        imp = imp.copy()
        imp["key"] = imp["smell"].map(TITLE_TO_KEY)
        for key, g in imp.groupby("key"):
            order = g["value"].abs().sort_values(ascending=False).index
            out["importance"][key] = _records(g.loc[order].head(12)[["feature", "value", "kind"]])
    return out


@lru_cache(maxsize=1)
def scores():
    """Frozen-model scores on the held-out test split (for curves and the threshold explorer)."""
    models, out = P.load_models(), {}
    for smell in SMELLS:
        art = models[smell]
        t = _dataset_csv(smell)
        t = t[t["split"] == "test"]
        p = art["model"].predict_proba(t[art["features"]].to_numpy(float))[:, 1]
        out[smell] = {"y": t["label"].astype(int).tolist(),
                      "p": [round(float(v), 4) for v in p],
                      "threshold": round(float(art["threshold"]), 4),
                      "model": art["model_name"]}
    return out


def tuning():
    df = _read(REPORTS / "feature_envy_tuning_cv.csv")
    return None if df is None else _records(df)


@lru_cache(maxsize=1)
def dataset():
    models = P.load_models()
    out = {"funnel": [], "files": None, "smells": {}, "repos": {}, "quality": {}, "features": {}}

    # ---- per-smell dataset statistics
    for smell in SMELLS:
        d = _dataset_csv(smell)
        tr, te, pos = d[d["split"] == "train"], d[d["split"] == "test"], d[d["label"] == 1]
        out["smells"][smell] = {
            "title": P.TITLES[smell], "level": P.SMELL_LEVEL[smell],
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "pos_train": int(tr["label"].sum()), "pos_test": int(te["label"].sum()),
            "prevalence": float(d["label"].mean()), "prevalence_test": float(te["label"].mean()),
            "agreement": {"unanimous": int((pos["label_mean"] >= 0.999).sum()),
                          "majority_only": int((pos["label_mean"] < 0.999).sum())},
            "reviews_per_location": {str(int(k)): int(v)
                                     for k, v in d["n_reviews"].value_counts().sort_index().items()},
        }

    # ---- repositories
    rs = _read(PROCESSED / "repo_split.csv")
    if rs is not None:
        out["repos"] = {"train": int((rs["split"] == "train").sum()),
                        "test": int((rs["split"] == "test").sum()), "total": int(len(rs))}

    # ---- feature counts
    out["features"] = {"class": len(models["blob"]["features"]),
                       "method": len(models["feature_envy"]["features"])}

    # ---- pipeline funnel and data-quality ledger
    raw = RAW / "MLCQCodeSmellSamples.csv"
    n_reviews = len(pd.read_csv(raw, sep=";", usecols=["id"])) if raw.exists() else None
    manifest = _read(RAW / "fetch_manifest.csv")
    if manifest is not None:
        out["files"] = {"unique": int(len(manifest)),
                        "fetched": int((manifest["status"] == "success").sum())}
    labeled = _read(INTERIM / "labeled_rows_with_files.csv", usecols=["id", "sample_id"])
    rep = _read(PROCESSED / "features_full_repaired.csv",
                usecols=["id", "smell", "label", "usable", "mapped", "match_source",
                         "repair_reason", "crlf_doubled"], low_memory=False)
    if rep is not None and labeled is not None:
        usable = rep[rep["usable"].astype(str) == "True"].merge(labeled, on="id", how="left")
        g = usable.groupby(["smell", "sample_id"])["label"].mean()
        ties = int((g == 0.5).sum())
        locations = int(len(g))
        final_rows = sum(v["n_train"] + v["n_test"] for v in out["smells"].values())
        mapped = int((rep["mapped"].astype(str) == "True").sum())
        n_usable = int(len(usable))
        crlf_rows = int((rep["crlf_doubled"].astype(str) == "True").sum())
        oom_rows = int((rep["match_source"] == "excluded_ck_oom").sum())
        repaired = int(rep["repair_reason"].notna().sum())
        scan = _read(INTERIM / "diagnostics" / "crlf_scan.csv")
        crlf_files = int((scan["n_doubled"] > 0).sum()) if scan is not None else None
        out["funnel"] = [
            {"label": "Human reviews in the MLCQ dataset", "count": n_reviews},
            {"label": "Reviews whose Java source was retrieved", "count": int(len(labeled))},
            {"label": "Reviews matched to code metrics", "count": mapped},
            {"label": "Reviews kept after quality exclusions", "count": n_usable},
            {"label": "Code locations after merging reviewers", "count": locations},
            {"label": "Training and test examples after dropping ties", "count": int(final_rows)},
        ]
        out["quality"] = {
            "reviews_total": n_reviews, "reviews_labeled": int(len(labeled)),
            "mapped": mapped, "unmapped": int(len(rep) - mapped),
            "repaired": repaired, "crlf_rows": crlf_rows, "crlf_files": crlf_files,
            "oom_rows": oom_rows, "usable": n_usable, "locations": locations,
            "ties_dropped": ties,
        }
    return out


@lru_cache(maxsize=1)
def reference():
    """Per smell: how the model's top metrics are distributed in the training split."""
    models, out = P.load_models(), {}
    for smell in SMELLS:
        d = _dataset_csv(smell)
        tr = d[d["split"] == "train"]
        out[smell] = {}
        for f in models[smell]["top_features"]:
            pos = tr[tr["label"] == 1][f].astype(float)
            neg = tr[tr["label"] == 0][f].astype(float)
            q = lambda s, p: round(float(s.quantile(p)), 3)  # noqa: E731
            out[smell][f] = {"pos_median": q(pos, 0.5), "pos_p25": q(pos, 0.25), "pos_p75": q(pos, 0.75),
                             "neg_median": q(neg, 0.5), "neg_p90": q(neg, 0.9)}
    return out


@lru_cache(maxsize=1)
def examples():
    """Held-out test cases chosen by rule (not by hand): per smell, the best-scored true
    positive, the lowest-scored missed positive, and the best-scored false alarm."""
    sc = scores()
    lab = pd.read_csv(INTERIM / "labeled_rows_with_files.csv",
                      usecols=["sample_id", "start_line", "end_line"]).drop_duplicates("sample_id")
    used, out = set(), []
    for smell in SMELLS:
        level = P.SMELL_LEVEL[smell]
        d = _dataset_csv(smell)
        t = d[d["split"] == "test"].reset_index(drop=True)
        t["p"] = sc[smell]["p"]
        thr = sc[smell]["threshold"]
        t = t.merge(lab, on="sample_id", how="left")
        picks = [("caught", t[(t["label"] == 1) & (t["p"] >= thr)].sort_values("p", ascending=False)),
                 ("missed", t[(t["label"] == 1) & (t["p"] < thr)].sort_values("p")),
                 ("false_alarm", t[(t["label"] == 0) & (t["p"] >= thr)].sort_values("p", ascending=False))]
        for outcome, cand in picks:
            for _, r in cand.iterrows():
                fid = int(r["file_id"])
                path = RAW / "source_files" / f"{fid}.java"
                if (fid in used or not path.exists() or path.stat().st_size > MAX_EXAMPLE_BYTES
                        or pd.isna(r["start_line"])):
                    continue
                used.add(fid)
                out.append({"id": f"{smell}-{outcome}", "smell": smell, "outcome": outcome, "level": level,
                            "file_id": fid, "code_name": r["code_name"],
                            "name": short_name(r["code_name"], level),
                            "score": round(float(r["p"]), 4), "threshold": thr,
                            "start_line": int(r["start_line"]), "end_line": int(r["end_line"]),
                            "label": int(r["label"]), "label_mean": round(float(r["label_mean"]), 2),
                            "n_reviews": int(r["n_reviews"])})
                break
    return out


def example_source(file_id):
    """Source text for one curated example (only files listed by examples() can be read)."""
    if int(file_id) not in {e["file_id"] for e in examples()}:
        return None
    path = RAW / "source_files" / f"{int(file_id)}.java"
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def warm():
    P.load_models()
    P.test_metrics()
    P.trained_class_types()
    for fn in (results, scores, dataset, reference, examples):
        fn()
