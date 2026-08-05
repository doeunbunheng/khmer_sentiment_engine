"""Pre-flight gate — run before any training/eval run to catch syntax and
import errors in seconds instead of after a long GPU run.

Checks:
  1. py_compile every .py under src/ and scripts/
  2. import every module under src/ (no model download, no GPU needed)
  3. report summary; exit code 1 on any failure

Usage:
  .venv\\Scripts\\python scripts\\preflight_check.py
"""

import importlib
import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"


def main():
    failures = []
    compiled = 0

    for base in (SRC, SCRIPTS):
        for py in sorted(base.rglob("*.py")):
            try:
                py_compile.compile(str(py), doraise=True)
                compiled += 1
            except py_compile.PyCompileError as e:
                failures.append(f"compile {py}: {e}")

    modules = []
    for py in sorted(SRC.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        rel = py.relative_to(PROJECT_ROOT).with_suffix("")
        modules.append(".".join(rel.parts))

    for mod in sorted(modules):
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures.append(f"import {mod}: {type(e).__name__}: {e}")

    print(f"preflight: compiled {compiled} files, imported {len(modules)} modules")
    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("OK — ready to run")


if __name__ == "__main__":
    main()
