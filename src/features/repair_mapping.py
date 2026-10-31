
"""
Stage 3.4 final repair pass. Evidence-only. Reads the diagnostic CSVs, writes
data/processed/features_full_repaired.csv. The unfiltered feature table is NOT modified.
"""

import pandas as pd

import map_ck_to_mlcq_full as m   # CK tables + helpers; main() is not run

ROOT = m.ROOT
PROCESSED = ROOT / "data" / "processed"
DIAG = ROOT / "data" / "interim" / "diagnostics"

orig = pd.read_csv(PROCESSED / "features_full_unfiltered.csv")
d1 = pd.read_csv(DIAG / "1_name_not_in_class.csv")
d2 = pd.read_csv(DIAG / "2_class_unresolved_function.csv")
d3 = pd.read_csv(DIAG / "3_class_unresolved_class.csv")
d4 = pd.read_csv(DIAG / "4_large_line_diff.csv")

# Identify source files affected by doubled CRLF line endings.
scan = pd.read_csv(DIAG / "crlf_scan.csv")
doubled_ids = set(
    scan.loc[scan["n_doubled"] > 0, "file_id"].astype(int)
)

feat = orig.copy()
ck_cols = [c for c in feat.columns if c.startswith("ck_")]
feat[ck_cols] = feat[ck_cols].astype(object)  # avoid dtype clashes when filling CK values
feat["repair_reason"] = ""
feat["quality_flag"] = "ok"
pos = {rid: i for i, rid in enumerate(feat["id"])}

changes, skipped = [], []


def is_target(rid, source):
    i = pos[rid]
    return (
        (not feat.at[i, "mapped"])
        and feat.at[i, "match_source"] == source
        and int(feat.at[i, "file_id"]) not in doubled_ids
    )


def apply(rid, hit, new_source, reason):
    i = pos[rid]
    old_source = feat.at[i, "match_source"]

    for k, v in m.ck_columns(hit).items():
        if k in feat.columns:
            feat.at[i, k] = v

    feat.at[i, "mapped"] = True
    feat.at[i, "match_source"] = new_source
    feat.at[i, "repair_reason"] = reason
    feat.at[i, "quality_flag"] = "repaired"

    if feat.at[i, "type"] == "function":
        feat.at[i, "line_diff"] = abs(
            int(hit["line"]) - feat.at[i, "start_line"]
        )

    changes.append({
        "id": rid,
        "type": feat.at[i, "type"],
        "old_source": old_source,
        "new_source": new_source,
        "reason": reason,
        "ck_class": feat.at[i, "ck_class"],
        "ck_method": (
            feat.at[i, "ck_method"]
            if feat.at[i, "type"] == "function"
            else ""
        ),
        "line_diff": feat.at[i, "line_diff"],
    })


def unique_nearest_method(fid, ck_classes, name, start):
    mdf = m.met_by[fid]
    cand = mdf[
        (mdf["class"].isin(ck_classes))
        & (mdf["mname"] == name)
    ]

    if cand.empty:
        return None

    d = (cand["line"] - start).abs()

    if d.min() > 2:
        return None

    near = cand[d == d.min()]
    return near.iloc[0] if len(near) == 1 else None


# ---- Repair 1: name_not_in_class -> same method name in exactly one other class at the line
for _, r in d1.iterrows():
    rid = r["id"]

    if (
        r["verdict"] != "same_name_in_other_class_at_line"
        or r["confidence"] != "high"
    ):
        skipped.append((
            rid, "name_not_in_class", r["verdict"], r["confidence"]
        ))
        continue

    if not is_target(rid, "name_not_in_class"):
        skipped.append((
            rid, "name_not_in_class", "not_in_expected_state", ""
        ))
        continue

    fid, start = int(r["file_id"]), r["start_line"]
    _, name, _ = m.parse_code_name(r["code_name"])
    mdf = m.met_by[fid]

    near_rows = mdf[
        (mdf["mname"] == name)
        & (mdf["class"] != r["ck_class"])
        & ((mdf["line"] - start).abs() <= 2)
    ]

    if near_rows["class"].nunique() != 1:
        skipped.append((
            rid, "name_not_in_class", "recheck_not_unique_class", ""
        ))
        continue

    hit = unique_nearest_method(
        fid, [near_rows["class"].iloc[0]], name, start
    )

    if hit is None:
        skipped.append((
            rid, "name_not_in_class", "recheck_not_unique_method", ""
        ))
        continue

    apply(
        rid,
        hit,
        "repaired_other_class_same_line",
        "same method name in exactly one other class of the file, "
        "line within 2 of MLCQ start_line",
    )


# ---- Repair 2: function rows with class_unresolved -> class resolved via $ or unique simple name
for _, r in d2.iterrows():
    rid = r["id"]

    ok_verdict = r["verdict"] in (
        "dollar_normalization",
        "simple_name_unique",
    )

    if not (ok_verdict and r["confidence"] == "high"):
        skipped.append((
            rid,
            "class_unresolved(function)",
            r["verdict"],
            r["confidence"],
        ))
        continue

    if not is_target(rid, "class_unresolved"):
        skipped.append((
            rid, "class_unresolved(function)", "not_in_expected_state", ""
        ))
        continue

    fid, start = int(r["file_id"]), r["start_line"]
    _, name, _ = m.parse_code_name(r["code_name"])

    hit = unique_nearest_method(
        fid, [r["target_ck_class"]], name, start
    )

    if hit is None:
        skipped.append((
            rid, "class_unresolved(function)", "recheck_failed", ""
        ))
        continue

    apply(
        rid,
        hit,
        "repaired_class_" + r["verdict"],
        f"class resolved via {r['verdict']}; "
        "method found within 2 lines of MLCQ start_line",
    )


# ---- Repair 3: class rows with class_unresolved
for _, r in d3.iterrows():
    rid = r["id"]

    ok_verdict = r["verdict"] in (
        "dollar_normalization",
        "simple_name_unique",
    )

    if not (ok_verdict and r["confidence"] == "high"):
        skipped.append((
            rid,
            "class_unresolved(class)",
            r["verdict"],
            r["confidence"],
        ))
        continue

    if not is_target(rid, "class_unresolved"):
        skipped.append((
            rid, "class_unresolved(class)", "not_in_expected_state", ""
        ))
        continue

    fid = int(r["file_id"])
    fc = m.cls_by[fid]
    hit = fc[fc["class"] == r["target_ck_class"]]

    if len(hit) != 1:
        skipped.append((
            rid, "class_unresolved(class)", "recheck_not_unique", ""
        ))
        continue

    apply(
        rid,
        hit.iloc[0],
        "repaired_class_" + r["verdict"],
        f"class resolved via {r['verdict']}; "
        "class declaration within 2 lines of MLCQ start_line",
    )


# ---- Quality flags for large line_diff rows (no remapping)
flag_map = {
    "likely_wrong_overload": "suspect_wrong_overload",
    "plausible_ck_line_inside_mlcq_range": "large_line_diff_plausible",
    "plausible_nearest_same_name_candidate": "large_line_diff_plausible",
    "unverified": "large_line_diff_unverified",
}

for _, r in d4.iterrows():
    feat.at[pos[r["id"]], "quality_flag"] = flag_map.get(
        r["verdict"], "large_line_diff_unverified"
    )


# ---- Exclude all rows belonging to CRLF-doubled source files
feat["crlf_doubled"] = feat["file_id"].isin(doubled_ids)
feat.loc[feat["crlf_doubled"], "quality_flag"] = "crlf_doubled"

feat["usable"] = (
    feat["mapped"]
    & ~feat["crlf_doubled"]
    & (feat["quality_flag"] != "suspect_wrong_overload")
)


# ---- Save repaired dataset and repair log
out_path = PROCESSED / "features_full_repaired.csv"
feat.to_csv(out_path, index=False)

log = pd.DataFrame(changes)
log.to_csv(DIAG / "repair_log.csv", index=False)


# ---- Validation: rows we did not touch must be identical to the original
rep = pd.read_csv(out_path)

changed_ids = set(log["id"]) if len(log) else set()

a = orig[~orig["id"].isin(changed_ids)].reset_index(drop=True)
b = rep[~rep["id"].isin(changed_ids)][list(orig.columns)].reset_index(drop=True)

try:
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    identical = True
except AssertionError as e:
    identical = False
    print("UNCHANGED-ROW CHECK FAILED:", str(e)[:400])


# ---- Report
n = len(orig)

print("=== Coverage ===")
print(
    f"original mapped: {int(orig['mapped'].sum())}/{n} "
    f"({100 * orig['mapped'].mean():.2f}%)"
)
print(
    f"repaired mapped: {int(rep['mapped'].sum())}/{n} "
    f"({100 * rep['mapped'].mean():.2f}%)"
)
print(f"usable for training (mapped, not suspect or CRLF-doubled): "
      f"{int(rep['usable'].sum())}")
print(
    f"rows changed: {len(log)} | "
    f"untouched rows identical to original: {identical}"
)

print("\nby type (repaired):")
print(
    rep.groupby("type")["mapped"]
    .agg(["sum", "count", "mean"])
    .round(4)
    .to_string()
)

# CRLF-doubled validation and usable-row counts
print(
    f"rows in CRLF-doubled files (excluded from training): "
    f"{int(rep['crlf_doubled'].sum())}"
)
print(
    "usable rows by smell:",
    rep[rep["usable"]].groupby("smell").size().to_dict()
)
print(
    "large line_diff (>50) among usable function rows:",
    int(
        (
            rep["usable"]
            & (rep["type"] == "function")
            & (rep["line_diff"] > 50)
        ).sum()
    ),
)

print("\n=== Changed rows by reason ===")
if len(log):
    print(log.groupby(["type", "new_source"]).size().to_string())

    for src, sub in log.groupby("new_source"):
        print(f"\n{src}: ids = {sorted(sub['id'].tolist())}")

    print("\nfirst 8 changes (full list in repair_log.csv):")
    print(log.head(8).to_string(index=False))


print("\n=== Candidates NOT applied (not clearly justified) ===")
sk = pd.DataFrame(
    skipped,
    columns=["id", "bucket", "verdict_or_reason", "confidence"],
)

if len(sk):
    print(
        sk.groupby(
            ["bucket", "verdict_or_reason", "confidence"]
        ).size().to_string()
    )
    print("ids:", sorted(sk["id"].tolist()))


print("\n=== Remaining unmapped ===")
print(
    rep[~rep["mapped"]]
    .groupby(["type", "match_source"])
    .size()
    .to_string()
)

print("\n=== Suspicious / flagged (mapped but flagged) ===")
print(rep["quality_flag"].value_counts().to_string())
print(
    "suspect_wrong_overload ids:",
    rep.loc[
        rep["quality_flag"] == "suspect_wrong_overload", "id"
    ].tolist(),
)

print(f"\nSaved: {out_path}")
