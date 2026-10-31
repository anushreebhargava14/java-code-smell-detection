from pathlib import Path
import pandas as pd

ROOT = Path("data/interim/ck_batches")
OUT = Path("data/interim/ck_full_output")
OUT.mkdir(parents=True, exist_ok=True)

class_parts = []
method_parts = []

for class_file in ROOT.rglob("class.csv"):
    # Ignore the final combined output if this script is run again
    if OUT in class_file.parents:
        continue

    try:
        df = pd.read_csv(class_file)
        if not df.empty:
            df["_ck_source"] = str(class_file)
            class_parts.append(df)
            print(f"CLASS OK: {class_file} -> {len(df)} rows")
    except Exception as e:
        print(f"CLASS ERROR: {class_file} -> {e}")

for method_file in ROOT.rglob("method.csv"):
    if OUT in method_file.parents:
        continue

    try:
        df = pd.read_csv(method_file)
        if not df.empty:
            df["_ck_source"] = str(method_file)
            method_parts.append(df)
            print(f"METHOD OK: {method_file} -> {len(df)} rows")
    except Exception as e:
        print(f"METHOD ERROR: {method_file} -> {e}")

if class_parts:
    classes = pd.concat(class_parts, ignore_index=True)
    classes.to_csv(OUT / "class.csv", index=False)
    print(f"\nCombined class rows: {len(classes)}")
    print(f"Unique class source files: {classes['file'].nunique()}")

if method_parts:
    methods = pd.concat(method_parts, ignore_index=True)
    methods.to_csv(OUT / "method.csv", index=False)
    print(f"Combined method rows: {len(methods)}")
    print(f"Unique method source files: {methods['file'].nunique()}")

print("\nDONE")