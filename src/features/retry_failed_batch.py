"""
Re-run only the failed batch 2 as smaller sub-batches, without touching
any other batch's input or output.
"""

import subprocess
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAILED_BATCH_INPUT = PROJECT_ROOT / "data" / "interim" / "ck_batches" / "batch_002" / "input"
SUBBATCH_ROOT = PROJECT_ROOT / "data" / "interim" / "ck_batches" / "batch_002_retry"
CK_JAR = PROJECT_ROOT / "tools" / "ck" / "target" / "ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar"

SUBBATCH_SIZE = 50
HEAP = "-Xmx3g"


def main():
    all_files = sorted(FAILED_BATCH_INPUT.glob("*.java"))
    n_sub = math.ceil(len(all_files) / SUBBATCH_SIZE)
    print(f"batch_002 retry: {len(all_files)} files -> {n_sub} sub-batches of {SUBBATCH_SIZE}")

    for i in range(n_sub):
        sub_files = all_files[i * SUBBATCH_SIZE:(i + 1) * SUBBATCH_SIZE]
        sub_input = SUBBATCH_ROOT / f"sub_{i:02d}" / "input"
        sub_output = SUBBATCH_ROOT / f"sub_{i:02d}" / "output"

        if (sub_output / "class.csv").exists():
            print(f"sub_{i:02d}: already done, skipping")
            continue

        sub_input.mkdir(parents=True, exist_ok=True)
        sub_output.mkdir(parents=True, exist_ok=True)

        for f in sub_files:
            link_path = sub_input / f.name
            if not link_path.exists():
                try:
                    link_path.hardlink_to(f)
                except (OSError, AttributeError):
                    link_path.write_bytes(f.read_bytes())

        cmd = ["java", HEAP, "-jar", str(CK_JAR), str(sub_input),
               "false", "0", "false", str(sub_output) + "\\"]
        print(f"sub_{i:02d}: running CK on {len(sub_files)} files...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"sub_{i:02d}: FAILED")
            print(result.stderr[-1000:])
        else:
            print(f"sub_{i:02d}: OK")


if __name__ == "__main__":
    main()