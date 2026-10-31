"""
Run CK in resumable batches to bound per-process memory usage.
Each batch gets its own output folder; a batch already producing
class.csv is skipped on rerun.
"""

import subprocess
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES_DIR = PROJECT_ROOT / "data" / "raw" / "source_files"
BATCH_ROOT = PROJECT_ROOT / "data" / "interim" / "ck_batches"
CK_JAR = PROJECT_ROOT / "tools" / "ck" / "target" / "ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar"

BATCH_SIZE = 300
HEAP = "-Xmx3g"


def main():
    all_files = sorted(SOURCE_FILES_DIR.glob("*.java"))
    n_batches = math.ceil(len(all_files) / BATCH_SIZE)
    print(f"{len(all_files)} files -> {n_batches} batches of up to {BATCH_SIZE}")

    for i in range(n_batches):
        batch_files = all_files[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_input_dir = BATCH_ROOT / f"batch_{i:03d}" / "input"
        batch_output_dir = BATCH_ROOT / f"batch_{i:03d}" / "output"

        if (batch_output_dir / "class.csv").exists():
            print(f"Batch {i}: already done, skipping")
            continue

        batch_input_dir.mkdir(parents=True, exist_ok=True)
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        # symlink/copy is avoided; instead point CK at a folder of hardlinks
        # to keep this fast and not duplicate 4000+ files on disk
        for f in batch_files:
            link_path = batch_input_dir / f.name
            if not link_path.exists():
                try:
                    link_path.hardlink_to(f)
                except (OSError, AttributeError):
                    link_path.write_bytes(f.read_bytes())

        cmd = ["java", HEAP, "-jar", str(CK_JAR), str(batch_input_dir),
               "false", "0", "false", str(batch_output_dir) + "\\"]
        print(f"Batch {i}: running CK on {len(batch_files)} files...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"Batch {i}: FAILED")
            print(result.stderr[-1000:])
        else:
            print(f"Batch {i}: OK")


if __name__ == "__main__":
    main()