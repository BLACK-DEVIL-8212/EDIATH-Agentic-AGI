#!/usr/bin/env python3
import os

root = '.'
skip_dirs = {'venv', '.venv', 'env', '__pycache__', '.git'}
count = 0
for dirpath, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if not f.endswith('.py'):
            continue
        path = os.path.join(dirpath, f)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                s = fh.read()
        except Exception:
            continue
        if 'except:' in s:
            ns = s.replace('\nexcept:\n', '\nexcept Exception:\n')
            if ns != s:
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write(ns)
                print('Patched', path)
                count += 1

print(f'Done. Patched {count} files')
