"""
Read-only check: do fetched source files contain doubled line endings (b"\r\r\n")?
Hypothesis: Stage 2 wrote files with Path.write_text() in text mode on Windows,
turning CRLF into "\r\r\n" so CK counts each line twice.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "source_files"
DIAG = ROOT / "data" / "interim" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

rows = []
for p in SRC.glob("*.java"):
    b = p.read_bytes()
    n_nl, n_dbl = b.count(b"\n"), b.count(b"\r\r\n")
    rows.append({"file_id": int(p.stem), "n_newlines": n_nl, "n_doubled": n_dbl,
                 "frac_doubled": n_dbl / n_nl if n_nl else 0.0})
files = pd.DataFrame(rows)
files.to_csv(DIAG / "crlf_scan.csv", index=False)

print("=== Source files ===")
print(f"total: {len(files)} | with any doubled endings: {int((files.n_doubled > 0).sum())} "
      f"| mostly doubled (>50% of lines): {int((files.frac_doubled > 0.5).sum())}")

feat = pd.read_csv(ROOT / "data" / "processed" / "features_full_unfiltered.csv")
lab = pd.read_csv(ROOT / "data" / "interim" / "labeled_rows_with_files.csv")[["id", "repository"]]
feat = feat.merge(lab, on="id", how="left").merge(
    files[["file_id", "n_doubled", "frac_doubled"]], on="file_id", how="left")
feat["in_doubled_file"] = feat["n_doubled"].fillna(0) > 0

print("\n=== MLCQ rows in files with doubled endings ===")
print(f"rows: {int(feat.in_doubled_file.sum())} of {len(feat)} "
      f"({100 * feat.in_doubled_file.mean():.1f}%) | repositories: "
      f"{feat[feat.in_doubled_file].repository.nunique()} of {feat.repository.nunique()}")
print(pd.crosstab(feat.in_doubled_file, [feat.type]).to_string())
print(feat[feat.in_doubled_file].groupby("smell").size().to_string())

fn = feat[(feat.type == "function") & feat.mapped & (feat.start_line >= 20)].copy()
fn["ratio"] = fn["ck_line"].astype(float) / fn["start_line"]
print("\n=== median CK line / MLCQ start_line (function rows, start_line >= 20) ===")
print(fn.groupby("in_doubled_file")["ratio"].median().round(2).to_string())
print("share of function rows with line_diff > 50:")
print(fn.groupby("in_doubled_file")["line_diff"].apply(lambda s: round((s > 50).mean(), 3)).to_string())

d4 = DIAG / "4_large_line_diff.csv"
if d4.exists():
    big_ids = set(pd.read_csv(d4)["id"])
    inb = feat[feat.id.isin(big_ids)]
    print(f"\nof the {len(big_ids)} rows with line_diff > 50, in doubled files: "
          f"{int(inb.in_doubled_file.sum())}")

print(f"\nSaved: {DIAG / 'crlf_scan.csv'}")