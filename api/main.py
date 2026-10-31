"""
FastAPI backend for the code smell detector: inference endpoints (Stage 8), read-only data
endpoints for the web UI (Stage 9), and the static frontend.

Run from the project root:  python -m uvicorn api.main:app --port 8000
"""

import shutil
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "inference"))
import predict as P  # noqa: E402
from api import insights as I  # noqa: E402

MAX_BYTES = 300_000
CK_SLOTS = threading.BoundedSemaphore(2)      # at most 2 CK/JVM runs at once

LIMITATIONS = [
    "Models were trained on the MLCQ dataset (Java, human-labeled) and are statistical: "
    "they flag likely smells, not certain ones.",
    "Feature Envy is the weakest model (about 1 in 10 flags is correct on held-out repositories) "
    "because the available metrics do not capture access to another class's data.",
    "Input must be a complete Java class or file; bare methods or fragments cannot be analyzed.",
    "Metrics are computed for the single file you provide; coupling metrics can differ slightly "
    "from training, where several files were analyzed together.",
]


@asynccontextmanager
async def lifespan(app):
    try:
        I.warm()
    except Exception as e:  # noqa: BLE001  (the API still starts if some reports are missing)
        print(f"[startup] could not preload all report data: {e!r}")
    yield


app = FastAPI(title="ML-Based Code Smell Detection", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    code: str


def _summary(res):
    tm, out = P.test_metrics(), {}
    for smell, title in P.TITLES.items():
        scored = [s for s in res["scores"] if s["smell"] == smell]
        out[smell] = {"title": title, "scored": len(scored),
                      "flagged": sum(1 for s in scored if s["flagged"]),
                      "test_precision": tm.get(smell, {}).get("precision"),
                      "test_recall": tm.get(smell, {}).get("recall")}
    return out


def _run(source: str) -> dict:
    if len(source.encode("utf-8")) > MAX_BYTES:
        raise HTTPException(413, f"Input too large (limit {MAX_BYTES // 1000} KB).")
    if not CK_SLOTS.acquire(timeout=30):
        raise HTTPException(503, "Server busy, please try again shortly.")
    try:
        res = P.analyze_source(source)
    finally:
        CK_SLOTS.release()
    if not res["ok"]:
        err = res["error"] or "Analysis failed."
        status = 500 if "Java was not found" in err else 504 if "timed out" in err else 422
        raise HTTPException(status, err)
    res["summary"] = _summary(res)
    return res


def _data(fn, *args):
    try:
        return fn(*args)
    except FileNotFoundError as e:
        raise HTTPException(404, f"Data not available yet: {Path(str(e.filename)).name}")


@app.get("/health")
def health():
    return {"status": "ok", "java": shutil.which("java") is not None,
            "ck_jar": P.CK_JAR.exists(), "models": sorted(P.load_models().keys())}


@app.get("/model-info")
def model_info():
    if not P.SUMMARY_CSV.exists():
        raise HTTPException(404, "Evaluation summary not found; run final_evaluation.py.")
    return {"models": pd.read_csv(P.SUMMARY_CSV).to_dict(orient="records"),
            "limitations": LIMITATIONS}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return _run(req.code)


@app.post("/analyze-file")
def analyze_file(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".java"):
        raise HTTPException(400, "Please upload a .java file.")
    raw = file.file.read(MAX_BYTES + 1)
    return _run(raw.decode("utf-8", errors="replace"))


# ---- read-only data endpoints for the web UI (all computed from real pipeline outputs)
@app.get("/api/results")
def api_results():
    return _data(I.results)


@app.get("/api/scores")
def api_scores():
    return _data(I.scores)


@app.get("/api/tuning")
def api_tuning():
    return {"rows": I.tuning()}


@app.get("/api/dataset")
def api_dataset():
    return _data(I.dataset)


@app.get("/api/reference")
def api_reference():
    return _data(I.reference)


@app.get("/api/examples")
def api_examples():
    return _data(I.examples)


@app.get("/api/example/{file_id}")
def api_example(file_id: int):
    src = _data(I.example_source, file_id)
    if src is None:
        raise HTTPException(404, "Unknown example.")
    return {"file_id": file_id, "source": src}


# ---- the frontend (must stay last so it never shadows an API route)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
