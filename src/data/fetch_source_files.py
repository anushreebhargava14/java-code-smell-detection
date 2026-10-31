"""
Stage 2: Fetch the Java source files referenced by MLCQ samples.

Approach: raw-content HTTP fetch per unique (repository, commit_hash, path),
at the pinned commit — not a full git clone. Deduplicated, resumable, rate-limited,
and fully logged to a manifest CSV.
"""

import csv
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SAMPLES_FILE = RAW_DATA_DIR / "MLCQCodeSmellSamples.csv"
SOURCE_FILES_DIR = RAW_DATA_DIR / "source_files"
MANIFEST_FILE = RAW_DATA_DIR / "fetch_manifest.csv"

REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def parse_owner_repo(repository: str) -> str:
    """
    Convert 'git@github.com:owner/repo.git' -> 'owner/repo'.
    """
    repo = repository.strip()
    repo = repo.replace("git@github.com:", "")
    repo = repo.replace("https://github.com/", "")
    if repo.endswith(".git"):
        repo = repo[:-4]
    return repo


def build_raw_url(repository: str, commit_hash: str, path: str) -> str:
    owner_repo = parse_owner_repo(repository)
    # MLCQ's 'path' column starts with a leading '/', raw URLs don't want a double slash
    clean_path = path.lstrip("/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{commit_hash}/{clean_path}"


def load_existing_manifest() -> dict:
    """
    Returns {(repository, commit_hash, path): status} for rows already processed,
    so a rerun can skip them.
    """
    if not MANIFEST_FILE.exists():
        return {}

    existing = {}
    with open(MANIFEST_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["repository"], row["commit_hash"], row["path"])
            existing[key] = row
    return existing


def fetch_with_retry(url: str) -> tuple[str, str]:
    """
    Returns (status, content_or_reason).
    status in {"success", "not_found", "error"}.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=15)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                return "error", str(e)
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue

        if response.status_code == 200:
            return "success", response.text
        if response.status_code == 404:
            return "not_found", "HTTP 404"
        if response.status_code in (429, 500, 502, 503):
            if attempt == MAX_RETRIES:
                return "error", f"HTTP {response.status_code} after retries"
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
        # Any other unexpected status
        return "error", f"HTTP {response.status_code}"

    return "error", "exhausted retries"


def main():
    SOURCE_FILES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SAMPLES_FILE, sep=";")
    unique_files = df[["repository", "commit_hash", "path"]].drop_duplicates().reset_index(drop=True)
    unique_files["file_id"] = unique_files.index

    print(f"Total unique files to process: {len(unique_files)}")

    existing = load_existing_manifest()
    print(f"Already in manifest from a previous run: {len(existing)}")

    manifest_rows = []
    already_done = 0
    newly_fetched = 0
    failed = 0

    for _, row in unique_files.iterrows():
        key = (row["repository"], row["commit_hash"], row["path"])
        file_id = row["file_id"]
        dest_path = SOURCE_FILES_DIR / f"{file_id}.java"

        if key in existing and existing[key]["status"] == "success" and dest_path.exists():
            manifest_rows.append(existing[key])
            already_done += 1
            continue

        url = build_raw_url(row["repository"], row["commit_hash"], row["path"])
        status, result = fetch_with_retry(url)

        if status == "success":
            dest_path.write_text(result, encoding="utf-8", errors="replace")
            newly_fetched += 1
        else:
            failed += 1
            print(f"  [{status}] file_id={file_id}: {result} ({url})")

        manifest_rows.append({
            "file_id": file_id,
            "repository": row["repository"],
            "commit_hash": row["commit_hash"],
            "path": row["path"],
            "status": status,
            "url": url,
        })

        time.sleep(REQUEST_DELAY_SECONDS)

    with open(MANIFEST_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "repository", "commit_hash", "path", "status", "url"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = len(unique_files)
    success_count = sum(1 for r in manifest_rows if r["status"] == "success")
    print("\n=== Summary ===")
    print(f"Total unique files:      {total}")
    print(f"Already done (skipped):  {already_done}")
    print(f"Newly fetched:           {newly_fetched}")
    print(f"Failed this run:         {failed}")
    print(f"Overall success rate:    {success_count}/{total} ({100 * success_count / total:.1f}%)")


if __name__ == "__main__":
    main()