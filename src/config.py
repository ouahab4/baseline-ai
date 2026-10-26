"""Project-wide paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Building Data Genome 2 (cleaned electricity meters), mirrored by NREL BuildingsBench on OEDI
DATA_URL = "https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v2.0.0/BuildingsBench/BDG-2"
SITES = ["Bear", "Fox", "Panther", "Rat"]
YEARS = [2016, 2017]

# Time-based split: models never see the test period during training
TRAIN_END = "2017-07-01"  # train: 2016-01-01 .. 2017-06-30, test: 2017-07-01 .. 2017-12-31

SEED = 42

for p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
