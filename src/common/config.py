from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

DATA = _cfg["data"]
PREPROCESSING = _cfg["preprocessing"]
DB = _cfg["db"]
MODEL = _cfg["model"]
ASPECT = _cfg["aspect_model"]
SEED = int(_cfg["project"]["seed"])

LABELED_DIR = PROJECT_ROOT / DATA["labeled_dir"]
SPLITS_DIR = PROJECT_ROOT / DATA["splits_dir"]
TRAIN_RATIO = float(DATA["train_ratio"])
VAL_RATIO = float(DATA["val_ratio"])
TEST_RATIO = float(DATA["test_ratio"])
ENCODING = DATA["encoding"]
