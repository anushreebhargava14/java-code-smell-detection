"""
Read-only: for the 62 rows repaired via 'repaired_other_class_same_line', check
parameter-signature compatibility (when MLCQ gave parameter types) and overload ambiguity.
"""

import pandas as pd

import map_ck_to_mlcq_full as m

ROOT = m.ROOT
rep = pd.read_csv(ROOT / "data" / "processed" / "features_full_repaired.csv", low_memory=False)
r62 = rep[rep["match_source"] == "repaired_other_class_same_line"]
print(f"repaired rows: {len(r62)}")

rows = []
for _, r in r62.iterrows():
    fid, start = int(r["file_id"]), r["start_line"]
    _, name, params = m.parse_code_name(r["code_name"])
    mdf = m.met_by[fid]
    same = mdf[(mdf["class"] == r["ck_class"]) & (mdf["mname"] == name)]
    near = same[(same["line"] - start).abs() <= 2]
    if params is None:
        sig = "no_params_in_code_name"
        status = ("unique_in_class" if len(same) == 1
                  else "overloads_but_one_within_2_lines" if len(near) == 1
                  else "AMBIGUOUS_overloads_near_line")
    else:
        want = [m.norm_type(p) for p in params]
        have = [m.norm_type(p) for p in m.ck_params(r["ck_method"])]
        sig = "params_given"
        status = ("verified_params_exact" if want == have
                  else "CONFLICT_param_types" if len(want) == len(have)
                  else "CONFLICT_param_arity")
    rows.append({"id": r["id"], "smell": r["smell"], "code_name": r["code_name"],
                 "start_line": start, "ck_class": r["ck_class"], "ck_method": r["ck_method"],
                 "line_diff": r["line_diff"], "n_same_name_in_class": len(same),
                 "n_within_2_lines": len(near), "mlcq_params": params, "sig": sig, "status": status})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "interim" / "diagnostics" / "repair_signature_check.csv", index=False)

print("distinct code locations among the 62 rows:",
      out[["code_name", "start_line"]].drop_duplicates().shape[0])
print("\n=== status counts ===")
print(out["status"].value_counts().to_string())
print("\n=== rows needing attention (CONFLICT / AMBIGUOUS) ===")
bad = out[out["status"].str.startswith(("CONFLICT", "AMBIGUOUS"))]
if len(bad):
    print(bad.assign(code_name=bad["code_name"].str.slice(0, 60))
          [["id", "code_name", "ck_method", "line_diff", "n_same_name_in_class",
            "n_within_2_lines", "mlcq_params", "status"]].to_string(index=False))
else:
    print("none")
print("\nsample of safe rows:")
print(out[~out.index.isin(bad.index)].head(5)
      [["id", "ck_method", "line_diff", "n_same_name_in_class", "mlcq_params", "status"]].to_string(index=False))