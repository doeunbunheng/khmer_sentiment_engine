# Project Notes

## Bash + Python one-liners with Khmer text (Windows MINGW64)

**Problem:** Running `python -c "..."` with Khmer/`!\n` text on Git Bash breaks.

Two separate failures:
1. `bash: !\n...: event not found` — `!` inside **double quotes** triggers bash *history expansion*.
2. `SyntaxError: unterminated string literal` — a **real newline** inside a string literal is invalid Python.

**Fix — single quotes outside, double quotes inside, use `\n` escapes:**
```bash
python -c 'from src.predict import predict; print(predict("កុំខកខាន...!\nប្រើ..."))'
```
Rules: `\` and `!` are only safe inside single quotes in bash. No raw newlines inside Python string literals.

## Printing Khmer output on Windows console

**Problem:** `UnicodeEncodeError: 'charmap' codec ... character maps to <undefined>` (cp1252 console).

**Fix:**
```bash
export PYTHONIOENCODING=utf-8   # or: $env:PYTHONIOENCODING='utf-8' in PowerShell
```

## UTF-8 source files

`data/splits/*.csv` are UTF-8. Khmer text in scripts: always read/write with `encoding="utf-8"`.

## Predicting Khmer text safely (preferred: scripts/predict_demo.py)

Avoid `python -c '...'` with Khmer entirely — quoting/encoding keeps breaking. Use the helper instead:
```bash
.venv\Scripts\python scripts\predict_demo.py notes.txt          # from a UTF-8 file
type notes.txt | .venv\Scripts\python scripts\predict_demo.py  # via stdin
.venv\Scripts\python scripts\predict_demo.py                    # paste mode (Ctrl+Z / Ctrl+D)
```
`predict_demo.py` reads UTF-8 from a file or stdin, so multi-line Khmer, `!`, and `\` are all safe.