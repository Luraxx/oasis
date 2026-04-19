"""
V5 Configuration — Paths, tiles, feature settings.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA = ROOT / "data" / "makeathon-challenge"
DATA_ADDITIONAL = ROOT / "data" / "sentinel-additional"
CACHE = ROOT / "cache" / "with2026"
MODELS = ROOT / "models" / "with2026"

# OOF paths
EKL_OOF = ROOT / "ekl_oof"
LUIS_V2_OOF = ROOT / "luis_v2_oof"

# Submission paths
EKL_SUBMISSION = ROOT / "ekl_submission"
LUIS_V4_SUBMISSION = ROOT / "luis_v4_submission"
SUBMISSION = ROOT / "submission"
SUBMISSION.mkdir(parents=True, exist_ok=True)
ARTIFACTS = ROOT / "artifacts"

# Tiles
TRAIN_TILES = [
    "18NWG_6_6", "18NWH_1_4", "18NWJ_8_9", "18NWM_9_4",
    "18NXH_6_8", "18NXJ_7_6", "18NYH_9_9", "19NBD_4_4",
    "47QMB_0_8", "47QQV_2_4",
    "48PUT_0_8", "48PWV_7_8", "48PXC_7_7", "48PYB_3_6",
    "48QVE_3_0", "48QWD_2_2",
]

TEST_TILES = [
    "18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6",
]

# Region mapping
def tile_region(tile: str) -> str:
    if tile.startswith("18N") or tile.startswith("19N"):
        return "amazon"
    elif tile.startswith("33N"):
        return "africa"
    else:
        return "asia"

REGIONS = {
    "amazon": [t for t in TRAIN_TILES if tile_region(t) == "amazon"],
    "asia":   [t for t in TRAIN_TILES if tile_region(t) == "asia"],
}

# Temporal
YEARS = list(range(2020, 2026))
MONTHS = list(range(1, 13))

# S2 indices for temporal analysis
S2_INDICES = ["ndvi", "nbr", "ndmi", "ndwi", "bsi", "evi"]
