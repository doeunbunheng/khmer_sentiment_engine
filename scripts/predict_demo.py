"""Predict sentiment from a UTF-8 file or stdin - avoids bash quoting issues with Khmer text.

Usage:
  python scripts/predict_demo.py path/to/text.txt
  type notes.txt | python scripts/predict_demo.py
  python scripts/predict_demo.py   # then paste text + Ctrl+Z (win) / Ctrl+D
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_sentiment


def read_input(argv):
    if len(argv) > 1:
        path = Path(argv[1])
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
        return path.read_text(encoding="utf-8")
    if sys.stdin.isatty():
        print("Paste text, then press Ctrl+Z Enter (Windows) or Ctrl+D (Unix):", file=sys.stderr)
    return sys.stdin.read()


def main():
    text = read_input(sys.argv)
    result = predict_sentiment(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()