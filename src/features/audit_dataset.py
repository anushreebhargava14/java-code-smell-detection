"""
Stage 4 prep: audit duplicates, repeated code locations, label conflicts, repo concentration.
Read-only.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
feat = pd.read_csv(ROOT / "data" / "processed" / "features_full_repaired.csv")
lab = pd.read_csv(ROOT / "data" / "interim" / "labeled_rows_with_files.csv")[
    ["id", "repository", "path", "sample_id", "reviewer_id", "is_from_industry_relevant_project"]]
df = feat.merge(lab, on="id", how="left")
use = df[df["usable"]].copy()
print(f"usable rows: {len(use)}\n")

print("=== A. Usable rows / positives per smell ===")
a = use.groupby("smell")["label"].agg(rows="count", positives="sum", pos_rate="mean").round(3)
print(a.to_string())

print("\n=== B. Reviews per code location (smell, sample_id) ===")
g = use.groupby(["smell", "sample_id"])
loc = g.agg(n_reviews=("id", "size"), n_reviewers=("reviewer_id", "nunique"),
            label_mean=("label", "mean"), label_nunique=("label", "nunique")).reset_index()
print("locations per smell:", loc.groupby("smell").size().to_dict())
print("reviews-per-location distribution:")
print(pd.crosstab(loc["smell"], loc["n_reviews"]).to_string())
print("locations with conflicting reviewer labels (present vs absent):")
print(loc.groupby("smell")["label_nunique"].apply(lambda s: int((s > 1).sum())).to_dict())

print("\n=== C. Same location under different sample_ids (repeated code) ===")
key = use.groupby(["file_id", "code_name", "smell"])["sample_id"].nunique()
print("(file, code_name, smell) keys with >1 sample_id:", int((key > 1).sum()))
key2 = use.groupby(["repository", "path", "start_line", "end_line", "smell"])["sample_id"].nunique()
print("(repo, path, lines, smell) keys with >1 sample_id:", int((key2 > 1).sum()))

print("\n=== D. Repository concentration (matters for the split) ===")
per_repo = use.groupby("repository").size().sort_values(ascending=False)
print(f"repositories: {len(per_repo)} | rows per repo median: {per_repo.median():.0f} | max: {per_repo.max()}")
print("top 8 repos share of rows:", (per_repo.head(8) / len(use)).round(3).to_dict())
for s, sub in use.groupby("smell"):
    pos_repos = sub[sub["label"] == 1]["repository"].nunique()
    print(f"{s}: repos with >=1 positive: {pos_repos} of {sub['repository'].nunique()}")

print("\n=== E. industry-relevance flag ===")
print(use.groupby("is_from_industry_relevant_project")["label"].agg(["count", "mean"]).round(3).to_string())

print("\n=== F. If aggregated to one row per (smell, sample_id), majority label ===")
loc["agg_label"] = (loc["label_mean"] > 0.5).astype(int)
loc["tie"] = loc["label_mean"] == 0.5
agg = loc.groupby("smell").agg(locations=("agg_label", "size"), positives=("agg_label", "sum"),
                                ties=("tie", "sum"))
print(agg.to_string())