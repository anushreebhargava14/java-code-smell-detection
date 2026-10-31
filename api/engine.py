"""
Scan engine. One or many Java files -> CK metrics in a single run (so coupling metrics see the whole
project, closer to how the models were trained) -> frozen models -> scored classes and methods.

Every flag also gets a model-agnostic "why": for each of the model's top metrics, how much the score
would drop if that one metric were typical of unflagged training code. This is computed with the real
model, not estimated.
"""

import io
import os
import re
import subprocess
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from api import insights as I
import predict as P

MAX_FILES = 150
MAX_FILE_BYTES = 300_000
MAX_TOTAL_BYTES = 3_000_000
NEAR_LIMIT = 40


class ScanError(Exception):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- input handling
def clean_name(name):
    parts = [p for p in PurePosixPath(str(name).replace("\\", "/")).parts if p not in ("/", "", ".", "..")]
    return "/".join(parts)


def collect(items):
    """items: list of (name, text). Returns (files dict, warnings)."""
    files, warnings = {}, []
    for name, text in items:
        name = clean_name(name)
        if not name.lower().endswith(".java"):
            continue
        size = len(text.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            warnings.append(f"{name} skipped: larger than {MAX_FILE_BYTES // 1000} KB")
            continue
        if not text.strip():
            continue
        base, i = name, 2
        while name in files:
            name = f"{base[:-5]}_{i}.java"
            i += 1
        files[name] = text.replace("\r\n", "\n")
    if not files:
        raise ScanError("No usable .java files were found.")
    if len(files) > MAX_FILES:
        raise ScanError(f"Too many files ({len(files)}). The limit is {MAX_FILES}; scan a smaller part of the project.")
    if sum(len(t.encode('utf-8')) for t in files.values()) > MAX_TOTAL_BYTES:
        raise ScanError(f"The files total more than {MAX_TOTAL_BYTES // 1_000_000} MB. Scan a smaller part of the project.")
    return files, warnings


def read_upload(filename, data):
    """One uploaded file -> list of (name, text). Zips are expanded safely (java files only)."""
    if filename.lower().endswith(".zip"):
        out = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                infos = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".java")]
                if len(infos) > 400:
                    raise ScanError("The ZIP holds more than 400 Java files. Scan a smaller part of the project.")
                if sum(i.file_size for i in infos) > 30_000_000:
                    raise ScanError("The ZIP expands to more than 30 MB of Java source.")
                for i in infos:
                    if i.file_size <= MAX_FILE_BYTES:
                        out.append((i.filename, z.read(i).decode("utf-8", errors="replace")))
        except zipfile.BadZipFile:
            raise ScanError("That file is not a valid ZIP archive.")
        return out
    return [(filename, data.decode("utf-8", errors="replace"))]


# ---------------------------------------------------------------- metrics
def run_ck_dir(files, heap="-Xmx2g", timeout=240):
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        out.mkdir()
        for name, text in files.items():
            p = src / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode("utf-8"))
        cmd = ["java", heap, "-jar", str(P.CK_JAR), str(src), "false", "0", "false", str(out) + os.sep]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise ScanError("Java was not found on PATH (needed to run the CK metrics tool).", 500)
        except subprocess.TimeoutExpired:
            raise ScanError("Metric extraction timed out. Try fewer or smaller files.", 504)
        if "OutOfMemoryError" in (proc.stderr or ""):
            raise ScanError("The metrics tool ran out of memory on this input. Try fewer files.", 422)
        return P._read_csv(out / "class.csv"), P._read_csv(out / "method.csv"), str(src)


def _relative(value, root, names):
    v = str(value).replace("\\", "/")
    r = root.replace("\\", "/").rstrip("/") + "/"
    if v.startswith(r):
        return v[len(r):]
    for n in sorted(names, key=len, reverse=True):
        if v == n or v.endswith("/" + n):
            return n
    return PurePosixPath(v).name


# ---------------------------------------------------------------- explanations
@lru_cache(maxsize=1)
def baselines():
    """Per smell: training medians of unflagged and of all rows, in model feature order."""
    out = {}
    for smell in P.SMELL_LEVEL:
        art = P.load_models()[smell]
        d = pd.read_csv(I.DATASET / f"{smell}.csv")
        tr = d[d["split"] == "train"]
        feats = art["features"]
        out[smell] = {"neg": tr[tr["label"] == 0][feats].median().to_numpy(float),
                      "all": tr[feats].median().to_numpy(float)}
    return out


def _num(v):
    v = float(v)
    return None if np.isnan(v) else round(v, 4)


def explain(smell, x):
    """Score for feature vector x (model feature order) and the per-metric score drop if typical."""
    art = P.load_models()[smell]
    feats = art["features"]
    base = baselines()[smell]["neg"]
    idx = [feats.index(f) for f in art["top_features"]]
    rows = [x.astype(float)]
    for i in idx:
        r = x.astype(float).copy()
        r[i] = base[i]
        rows.append(r)
    p = art["model"].predict_proba(np.vstack(rows))[:, 1]
    why = [{"feature": feats[i], "value": _num(x[i]), "typical": _num(base[i]),
            "drop": round(float(p[0] - p[k + 1]), 4)} for k, i in enumerate(idx)]
    why.sort(key=lambda w: -w["drop"])
    return float(p[0]), why


# ---------------------------------------------------------------- scan
def scan(files):
    names = list(files)
    classes, methods, root = run_ck_dir(files)
    warnings = []
    allowed = P.trained_class_types()
    if allowed and "type" in classes.columns:
        classes = classes[classes["type"].astype(str).isin(allowed)]
    classes, methods = classes.reset_index(drop=True), methods.reset_index(drop=True)
    if classes.empty and methods.empty:
        raise ScanError("No classes or methods could be analyzed. Provide complete Java classes with valid syntax, "
                        "not bare methods or fragments.")
    for df in (classes, methods):
        if len(df):
            df["__rel"] = df["file"].map(lambda v: _relative(v, root, names))

    models, tm = P.load_models(), P.test_metrics()
    findings, near, counts = [], [], {}
    per_file = {n: {"classes": 0, "methods": 0, "flagged": 0, "max_ratio": 0.0, "top_smell": None} for n in names}
    for _, r in classes.iterrows():
        if r["__rel"] in per_file:
            per_file[r["__rel"]]["classes"] += 1
    for _, r in methods.iterrows():
        if r["__rel"] in per_file:
            per_file[r["__rel"]]["methods"] += 1

    for smell, level in P.SMELL_LEVEL.items():
        df = classes if level == "class" else methods
        art = models[smell]
        counts[smell] = {"scored": int(len(df)), "flagged": 0,
                         "test_precision": tm.get(smell, {}).get("precision"),
                         "test_recall": tm.get(smell, {}).get("recall")}
        if df.empty:
            continue
        X = P.feature_matrix(df, art["features"], warnings)
        proba = art["model"].predict_proba(X)[:, 1]
        thr = float(art["threshold"])
        for i in range(len(df)):
            row, rel, p = df.iloc[i], df.iloc[i]["__rel"], float(proba[i])
            if level == "class":
                name, line, ck_method = str(row["class"]), P.class_line(files.get(rel, ""), row["class"]), None
            else:
                name = P.method_display(row["class"], row["method"])
                line, ck_method = (int(row["line"]) if pd.notna(row["line"]) else None), str(row["method"])
            ratio = p / thr if thr else 0.0
            entry = {"file": rel, "smell": smell, "level": level, "name": name, "line": line,
                     "ck_class": str(row["class"]), "ck_method": ck_method,
                     "score": round(p, 4), "threshold": round(thr, 4), "ratio": round(ratio, 3),
                     "flagged": bool(p >= thr)}
            pf = per_file.get(rel)
            if pf is not None and ratio > pf["max_ratio"]:
                pf["max_ratio"], pf["top_smell"] = round(ratio, 3), smell
            if entry["flagged"]:
                counts[smell]["flagged"] += 1
                if pf is not None:
                    pf["flagged"] += 1
                _, entry["why"] = explain(smell, X[i])
                entry["key_metrics"] = {f: _num(row[f]) for f in art["top_features"] if f in df.columns}
                findings.append(entry)
            elif ratio >= 0.75:
                near.append(entry)
    findings.sort(key=lambda f: -f["ratio"])
    near.sort(key=lambda f: -f["ratio"])
    file_rows = [{"name": n, "lines": files[n].count("\n") + 1, **per_file[n],
                  "parsed": (per_file[n]["classes"] + per_file[n]["methods"]) > 0} for n in names]
    unparsed = [f["name"] for f in file_rows if not f["parsed"]]
    if unparsed:
        warnings.append(f"{len(unparsed)} file(s) could not be measured (syntax the metrics tool could not read): "
                        + ", ".join(unparsed[:5]) + ("…" if len(unparsed) > 5 else ""))
    return {"files": file_rows, "findings": findings, "near": near[:NEAR_LIMIT], "counts": counts,
            "n_classes": int(len(classes)), "n_methods": int(len(methods)),
            "warnings": sorted(set(warnings))}


def diff(parent, new):
    """Compare two runs by (file, smell, entity name)."""
    key = lambda f: (f["file"], f["smell"], f["name"])  # noqa: E731
    old = {key(f): f for f in parent["findings"]}
    cur = {key(f): f for f in new["findings"]}
    return {
        "parent": parent["id"],
        "resolved": [{"smell": f["smell"], "name": f["name"], "file": f["file"], "score": f["score"]}
                     for k, f in old.items() if k not in cur],
        "new": [{"smell": f["smell"], "name": f["name"], "file": f["file"], "score": f["score"], "line": f["line"]}
                for k, f in cur.items() if k not in old],
        "changed": [{"smell": f["smell"], "name": f["name"], "file": f["file"], "line": f["line"],
                     "before": old[k]["score"], "after": f["score"]}
                    for k, f in cur.items() if k in old and abs(f["score"] - old[k]["score"]) >= 0.005],
        "flagged_before": len(old), "flagged_after": len(cur),
    }


# ---------------------------------------------------------------- model lab
def lab_score(level, values, case=None):
    """Score a hypothetical (or real, when `case` is given) vector with the real models."""
    out, tm = [], P.test_metrics()
    for smell in I.LAB_LEVELS[level]:
        art = P.load_models()[smell]
        feats = art["features"]
        vec = None
        if case:
            vec = I.case_vector(case["smell"], case["sample_id"], feats)
        if vec is None:
            vec = baselines()[smell]["neg"]
        vec = np.array(vec, dtype=float)   # writable copy
        for k, v in (values or {}).items():
            if k in feats:
                vec[feats.index(k)] = float(v)
        score, why = explain(smell, vec)
        thr = float(art["threshold"])
        out.append({"smell": smell, "score": round(score, 4), "threshold": round(thr, 4),
                    "flagged": bool(score >= thr), "why": why,
                    "test_precision": tm.get(smell, {}).get("precision")})
    return out


def lab_case_values(level, smell, sample_id):
    """Slider values taken from a real held-out case."""
    spec = I.lab_spec()[level]
    names = [f["name"] for f in spec["features"]]
    vec = I.case_vector(smell, sample_id, names)
    return None if vec is None else {n: round(float(v), 2) for n, v in zip(names, vec)}
