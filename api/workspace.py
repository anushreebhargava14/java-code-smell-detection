"""Saved scans ("runs"), stored as small JSON files under data/workspace/. Nothing leaves this machine."""

import json
import re
import secrets
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data" / "workspace"
MAX_RUNS = 60
ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def new_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def valid(run_id):
    return bool(ID_RE.match(str(run_id)))


def _write(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def save(run, sources):
    WS.mkdir(parents=True, exist_ok=True)
    rid = run["id"]
    _write(WS / f"{rid}.json", run)
    _write(WS / f"{rid}.src.json", sources)
    strip = [round(f["max_ratio"], 2) for f in run["files"]][:80]
    meta = {"id": rid, "created": run["created"], "name": run["name"], "kind": run["kind"],
            "n_files": len(run["files"]), "n_findings": len(run["findings"]),
            "flagged": {k: v["flagged"] for k, v in run["counts"].items()},
            "parent": run.get("parent"), "strip": strip, "case": bool(run.get("case"))}
    _write(WS / f"{rid}.meta.json", meta)
    metas = sorted(WS.glob("*.meta.json"))
    for old in metas[:-MAX_RUNS]:
        delete(old.name.replace(".meta.json", ""))
    return meta


def list_runs():
    if not WS.exists():
        return []
    out = []
    for p in sorted(WS.glob("*.meta.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def load(run_id):
    if not valid(run_id):
        return None
    p = WS / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_sources(run_id):
    if not valid(run_id):
        return None
    p = WS / f"{run_id}.src.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def delete(run_id):
    if not valid(run_id):
        return False
    found = False
    for suffix in (".json", ".src.json", ".meta.json"):
        p = WS / f"{run_id}{suffix}"
        if p.exists():
            p.unlink()
            found = True
    return found
