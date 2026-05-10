#!/usr/bin/env python3
"""Walk the repo and check Python syntax for all .py files.
Prints any SyntaxErrors found and exits with non-zero code if any.
"""

import os
import sys

skip_dirs = {".git", "__pycache__", "node_modules", "venv", "env", "build", "dist"}
errors = []

for root, dirs, files in os.walk("."):
    # skip virtual env and other directories
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            compile(src, path, "exec")
        except SyntaxError as e:
            errors.append((path, e))
        except Exception:
            # ignore other runtime issues when reading non-text files
            continue

if errors:
    print(f"Found {len(errors)} file(s) with SyntaxError:\n")
    for path, e in errors:
        print(f"File: {path}")
        print(f"  {e.__class__.__name__}: {e.msg} (line {e.lineno})")
        if e.text:
            print(f"    {e.text.strip()}")
        print()
    sys.exit(2)

print("No Python syntax errors found.")
sys.exit(0)
