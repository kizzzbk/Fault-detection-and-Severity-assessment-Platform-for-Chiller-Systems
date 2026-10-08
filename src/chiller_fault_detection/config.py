"""Project paths and loader constants.

Path *strings* live here. Thin helpers in ``load`` (or below) turn them into
absolute ``Path`` objects relative to the repo root so CLI / notebooks work
regardless of the current working directory.
"""

from __future__ import annotations
from pathlib import Path

# Package: src/chiller_fault_detection/config.py → parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data" / "ASHRAE RP-1043" / "Chiller Data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
DATASET_PARQUET = PROCESSED_DIR / "dataset.parquet"
DATASET_META_JSON = PROCESSED_DIR / "dataset.meta.json"

SPLITS_DIR = PROCESSED_DIR / "splits"
SPLIT_INFO_JSON = SPLITS_DIR / "split_info.json"
SCALER_PATH = SPLITS_DIR / "scaler.joblib"
RESIDUALIZER_PATH = SPLITS_DIR / "residualizer.joblib"

# Row filter before split: chiller on and kW physically plausible.
KW_MIN = 10.0
KW_MAX = 200.0

SHEET_NAME = "Complete Data Set"
TIME_COLUMN = "Time"
TINY_ABS = 1e-20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
