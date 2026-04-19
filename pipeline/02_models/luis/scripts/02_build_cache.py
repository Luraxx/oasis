#!/usr/bin/env python3
"""
Step 1: Build feature cache for all tiles.

Usage:
    python scripts/01_build_cache.py              # all tiles
    python scripts/01_build_cache.py 18NWG_6_6    # single tile
"""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAIN_TILES, TEST_TILES
from src.features.pack import save_tile


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    all_tiles = TRAIN_TILES + TEST_TILES

    for tile in all_tiles:
        split = "train" if tile in TRAIN_TILES else "test"
        if target != "all" and tile != target:
            continue
        print(f"\n{'=' * 60}")
        print(f"Processing {tile} ({split})")
        save_tile(tile, split)

    print("\nAll tiles processed!")


if __name__ == "__main__":
    main()
