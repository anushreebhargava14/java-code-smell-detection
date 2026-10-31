"""
Stage 7: inference pipeline. Java source text -> CK metrics -> trained models -> smell predictions.
Frozen train-split models from models/*.joblib are used as-is.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CK_JAR = ROOT / "tools" / "ck" / "target" / "ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar"
MODELS_DIR = ROOT / "models"
SUMMARY_CSV = ROOT / "reports" / "final_evaluation_summary.csv"
TRAIN_TABLE = ROOT / "data" / "processed" / "features_full_repaired.csv"

SMELL_LEVEL = {"blob": "class", "data_class": "class",
               "feature_envy": "method", "long_method": "method"}
TITLES = {"blob": "Blob (God Class)", "data_class": "Data Class",
          "feature_envy": "Feature Envy", "long_method": "Long Method"}
BOOL_MAP = {"True": 1, "False": 0, True: 1, False: 0}


@lru_cache(maxsize=1)
def load_models():
    models = {}
    for smell in SMELL_LEVEL:
        art = joblib.load(MODELS_DIR / f"{smell}.joblib")
        clf = art["model"].named_steps["clf"]
        weights = clf.feature_importances_ if hasattr(clf, "feature_importances_") else clf.coef_[0]
        order = pd.Series(np.abs(weights), index=art["features"]).sort_values(ascending=False)
        art["top_features"] = list(order.index[:5])
        models[smell] = art
    return models


@lru_cache(maxsize=1)
def test_metrics():
    if not SUMMARY_CSV.exists():
        return {}
    s = pd.read_csv(SUMMARY_CSV).set_index("smell")
    return {k: {"precision": float(s.loc[TITLES[k], "precision"]),
                "recall": float(s.loc[TITLES[k], "recall"])}
            for k in SMELL_LEVEL if TITLES[k] in s.index}


@lru_cache(maxsize=1)
def trained_class_types():
    """CK 'type' values of the class rows the class models were trained on (None = unknown)."""
    if not TRAIN_TABLE.exists():
        return None
    d = pd.read_csv(TRAIN_TABLE, usecols=["type", "usable", "ck_type"], low_memory=False)
    d = d[(d["type"] == "class") & (d["usable"].astype(str) == "True")]
    return set(d["ck_type"].dropna().astype(str)) or None


def _read_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_ck(source, heap="-Xmx1g", timeout=180):
    with tempfile.TemporaryDirectory() as tmp:
        src_dir, out_dir = Path(tmp) / "src", Path(tmp) / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        # raw bytes, no newline translation
        (src_dir / "Input.java").write_bytes(source.replace("\r\n", "\n").encode("utf-8"))
        cmd = ["java", heap, "-jar", str(CK_JAR), str(src_dir), "false", "0", "false",
               str(out_dir) + os.sep]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return _read_csv(out_dir / "class.csv"), _read_csv(out_dir / "method.csv"), proc


def feature_matrix(df, features, warnings):
    cols, missing = {}, [f for f in features if f not in df.columns]
    if missing:
        warnings.append(f"CK output lacks metrics {missing}; imputed with training medians")
    for f in features:
        if f in df.columns:
            s = df[f]
            if not pd.api.types.is_numeric_dtype(s):
                s = s.replace(BOOL_MAP)
            cols[f] = pd.to_numeric(s, errors="coerce")
        else:
            cols[f] = pd.Series(np.nan, index=df.index)
    X = pd.DataFrame(cols)[features].to_numpy(float)
    if (X < 0).any():
        warnings.append("negative metric values clipped to 0")
        X = np.clip(X, 0, None)
    return X


def _num(v):
    try:
        return None if pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def method_display(cls, method):
    simple = str(cls).replace("$", ".").split(".")[-1]
    name, _, rest = str(method).partition("/")
    args = rest.split("[", 1)[1].rstrip("]") if "[" in rest else ""
    return f"{simple}.{name}({args})"


def class_line(source, class_name):
    simple = str(class_name).replace("$", ".").split(".")[-1]
    rx = re.compile(rf"\b(?:class|interface|enum)\s+{re.escape(simple)}\b")
    for i, line in enumerate(source.replace("\r\n", "\n").split("\n"), start=1):
        if rx.search(line):
            return i
    return None


def analyze_source(source):
    res = {"ok": False, "error": None, "warnings": [], "n_classes": 0, "n_methods": 0,
           "scores": [], "findings": []}
    if not source or not source.strip():
        res["error"] = "Empty input."
        return res
    try:
        classes, methods, _ = run_ck(source)
    except FileNotFoundError:
        res["error"] = "Java was not found on PATH (needed to run the CK metrics tool)."
        return res
    except subprocess.TimeoutExpired:
        res["error"] = "Metric extraction timed out."
        return res

    allowed = trained_class_types()
    if allowed and "type" in classes.columns:
        skipped = classes[~classes["type"].astype(str).isin(allowed)]
        if len(skipped):
            res["warnings"].append("skipped class-level entities of unsupported kind: "
                                   + ", ".join(sorted(set(skipped["type"].astype(str)))))
        classes = classes[classes["type"].astype(str).isin(allowed)]
    classes, methods = classes.reset_index(drop=True), methods.reset_index(drop=True)
    res["n_classes"], res["n_methods"] = len(classes), len(methods)
    if classes.empty and methods.empty:
        res["error"] = ("No classes or methods could be analyzed. Provide a complete Java class "
                        "(or file) with valid syntax, not a bare method or fragment.")
        return res

    models, tm = load_models(), test_metrics()
    for smell, level in SMELL_LEVEL.items():
        df = classes if level == "class" else methods
        if df.empty:
            continue
        art = models[smell]
        proba = art["model"].predict_proba(feature_matrix(df, art["features"], res["warnings"]))[:, 1]
        thr = float(art["threshold"])
        for i in range(len(df)):
            row = df.iloc[i]
            if level == "class":
                name, line, ck_method = str(row["class"]), class_line(source, row["class"]), None
            else:
                name = method_display(row["class"], row["method"])
                line, ck_method = (int(row["line"]) if pd.notna(row["line"]) else None), str(row["method"])
            p = float(proba[i])
            res["scores"].append({
                "smell": smell, "title": TITLES[smell], "level": level, "name": name, "line": line,
                "ck_class": str(row["class"]), "ck_method": ck_method,
                "probability": round(p, 4), "threshold": round(thr, 4), "flagged": bool(p >= thr),
                "model": art["model_name"],
                "key_metrics": {f: _num(row[f]) for f in art["top_features"] if f in df.columns},
                "test_precision": tm.get(smell, {}).get("precision"),
                "test_recall": tm.get(smell, {}).get("recall")})
    res["findings"] = sorted([s for s in res["scores"] if s["flagged"]], key=lambda s: -s["probability"])
    res["ok"] = True
    return res


def print_report(res):
    if not res["ok"]:
        print("ERROR:", res["error"])
        return
    print(f"classes scored: {res['n_classes']} | methods scored: {res['n_methods']}")
    for w in sorted(set(res["warnings"])):
        print("warning:", w)
    tm = test_metrics()
    for smell in SMELL_LEVEL:
        scored = [s for s in res["scores"] if s["smell"] == smell]
        flagged = sorted([s for s in scored if s["flagged"]], key=lambda s: -s["probability"])
        note = (f"  [held-out test: precision {tm[smell]['precision']:.0%}, "
                f"recall {tm[smell]['recall']:.0%}]") if smell in tm else ""
        print(f"\n{TITLES[smell]}: {len(flagged)} flagged of {len(scored)}{note}")
        for s in flagged[:10]:
            km = ", ".join(f"{k}={v:g}" for k, v in s["key_metrics"].items() if v is not None)
            print(f"  line {s['line']}: {s['name']}  p={s['probability']:.2f} "
                  f"(threshold {s['threshold']:.2f}) | {km}")
        if len(flagged) > 10:
            print(f"  ... and {len(flagged) - 10} more")


def main():
    ap = argparse.ArgumentParser(description="Detect code smells in a Java file.")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()
    res = analyze_source(Path(args.path).read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(res, indent=2) if args.json else "", end="")
    if not args.json:
        print_report(res)


if __name__ == "__main__":
    main()