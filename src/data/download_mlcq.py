"""
Stage 1, Step 1: Download the MLCQ dataset from Zenodo.

Source: Madeyski & Lewowski (2020), MLCQ: Industry-relevant code smell data set.
DOI: 10.5281/zenodo.3666840, License: CC BY 4.0
"""

import requests
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

FILES_TO_DOWNLOAD = {
    "MLCQCodeSmellSamples.csv":
        "https://zenodo.org/records/3666840/files/MLCQCodeSmellSamples.csv?download=1",
    "MLCQCodeSmellDevelopersSurvey.csv":
        "https://zenodo.org/records/3666840/files/MLCQCodeSmellDevelopersSurvey.csv?download=1",
    "MadeyskiLewowskiMLCQAppendix.pdf":
        "https://zenodo.org/records/3666840/files/MadeyskiLewowskiMLCQAppendix.pdf?download=1",
}


def download_file(url: str, destination: Path):
    if destination.exists():
        print(f"Already exists, skipping: {destination.name}")
        return

    print(f"Downloading {destination.name} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = destination.stat().st_size / 1024
    print(f"  Saved: {destination} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES_TO_DOWNLOAD.items():
        download_file(url, RAW_DATA_DIR / filename)

    print("\nDone. Files in data/raw/:")
    for f in sorted(RAW_DATA_DIR.iterdir()):
        print(f"  {f.name}")