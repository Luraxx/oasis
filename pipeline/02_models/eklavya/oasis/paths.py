"""Canonical filesystem paths for the oasis pipeline.

Every other module imports these and never hardcodes paths.

Override the cache root with ``OASIS_CACHE_ROOT`` (defaults to
``data/cache`` next to the dataset; on the MI300X box you can point
this at the 5 TB scratch SSD).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = REPO_ROOT / "data" / "makeathon-challenge"
S2_DIR = DATA_ROOT / "sentinel-2"
S1_DIR = DATA_ROOT / "sentinel-1"
AEF_DIR = DATA_ROOT / "aef-embeddings"
LABEL_DIR = DATA_ROOT / "labels" / "train"
META_DIR = DATA_ROOT / "metadata"

CACHE_ROOT = Path(os.environ.get("OASIS_CACHE_ROOT", REPO_ROOT / "data" / "cache"))
EXTERNALS_ROOT = Path(os.environ.get("OASIS_EXTERNALS_ROOT", REPO_ROOT / "data" / "externals"))

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
MODELS_ROOT = ARTIFACTS_ROOT / "models"
OOF_ROOT = ARTIFACTS_ROOT / "oof"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"

SUBMISSION_ROOT = REPO_ROOT / "submission"

LABEL_SOURCES = ("radd", "gladl", "glads2")


def ensure_dirs() -> None:
    """Create artifact and cache directories if missing."""
    for path in (CACHE_ROOT, EXTERNALS_ROOT, ARTIFACTS_ROOT, MODELS_ROOT, OOF_ROOT, LOGS_ROOT, SUBMISSION_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def tile_cache_dir(tile_id: str, split: str) -> Path:
    """Per-tile cache directory."""
    return CACHE_ROOT / split / tile_id


def s2_tile_dir(tile_id: str, split: str) -> Path:
    return S2_DIR / split / f"{tile_id}__s2_l2a"


def s1_tile_dir(tile_id: str, split: str) -> Path:
    return S1_DIR / split / f"{tile_id}__s1_rtc"


def aef_tile_files(tile_id: str, split: str) -> list[Path]:
    split_dir = AEF_DIR / split
    return sorted(split_dir.glob(f"{tile_id}_*.tiff"))
