"""Generate per-tile probability rasters from lgbm_cmr model.

Format matches the existing ekl/luis prob rasters: single-band uint16 with
values 0-1000 (= prob 0.000-1.000). One file per test tile in canonical UTM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import rasterio

ROOT = Path("/shared-docker/oasis-mark-2")
sys.path.insert(0, str(ROOT))

from src.data.canonical_grid import all_tile_ids, grid_for  # noqa
from src.model.dataset import load_tile  # noqa

OUT = ROOT / "external" / "cmr_prob"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_cmr.txt"))
    test_tiles = sorted(t for t in all_tile_ids() if grid_for(t).split == "test")
    for tile in test_tiles:
        grid = grid_for(tile)
        bundle = load_tile(tile, sample_neg_ratio=None)
        H, W = bundle.shape
        probs = booster.predict(bundle.X)
        pred_map = np.zeros(H * W, dtype=np.float32)
        pred_map[bundle.pixel_index] = probs
        pred_map = pred_map.reshape(H, W)
        pred_u16 = (pred_map * 1000).clip(0, 65535).astype(np.uint16)

        profile = grid.rasterio_profile(count=1, dtype="uint16", nodata=0)
        out = OUT / f"prob_{tile}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(pred_u16, 1)
        print(f"  {tile}: shape={pred_u16.shape} max={pred_u16.max()}/1000  "
              f"pos@0.40={(pred_u16 > 400).sum():,}  pos@0.20={(pred_u16 > 200).sum():,}  → {out}")


if __name__ == "__main__":
    main()
