"""
Stage 0 sanity check.
Confirms Python, pip packages, and Java are all correctly set up
before we build anything on top of this environment.
"""

import subprocess
import sys


def check_python_version():
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major != 3 or version.minor < 9:
        print("  WARNING: Python 3.9+ recommended.")
    else:
        print("  OK")


def check_package(package_name):
    try:
        __import__(package_name)
        print(f"Package '{package_name}': OK")
    except ImportError:
        print(f"Package '{package_name}': MISSING (run: pip install {package_name})")


def check_java():
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
        )
        output = result.stderr or result.stdout
        first_line = output.strip().split("\n")[0]
        print(f"Java: {first_line}")
        print("  OK")
    except FileNotFoundError:
        print("Java: NOT FOUND")
        print("  You will need Java (JDK 8+) installed before Stage 2.")


def check_git():
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
        )
        print(f"Git: {result.stdout.strip()}")
        print("  OK")
    except FileNotFoundError:
        print("Git: NOT FOUND")


if __name__ == "__main__":
    print("=== Environment Check ===\n")
    check_python_version()
    print()
    check_package("requests")
    check_package("dotenv")
    print()
    check_java()
    print()
    check_git()
    print("\n=== Done ===")