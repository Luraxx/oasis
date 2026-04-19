"""Crop Hansen treecover2000/lossyear/datamask tiles for CMR tiles.

Source: external/hansen/{treecover2000,lossyear,datamask}_10N_010E.tif
Output: external/hansen/cropped/{tile}_{layer}.tif

We crop a bbox slightly larger than the canonical tile footprint so that
reproject_to_canonical has some buffer, then reproject-on-read at feature time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

ROOT = Path("/shared-docker/oasis-mark-2")
sys.path.insert(0, str(ROOT))

from src.data.canonical_grid import grid_for, all_tile_ids  # noqa

HANSEN_SRC = ROOT / "external/hansen"
HANSEN_DST = ROOT / "external/hansen/cropped"
HANSEN_DST.mkdir(parents=True, exist_ok=True)
LAYERS = ["treecover2000", "lossyear", "datamask"]
SRC_TILE = "10N_010E"  # covers 0-10N, 10-20E (southern Cameroon)


def crop_one(tile: str):
    grid = grid_for(tile)
    # Bounds of canonical tile in WGS84 (Hansen is EPSG:4326)
    w, s, e, n = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)
    # expand by ~0.01° (~1 km) buffer
    w -= 0.01; s -= 0.01; e += 0.01; n += 0.01

    for layer in LAYERS:
        src_path = HANSEN_SRC / f"{layer}_{SRC_TILE}.tif"
        dst_path = HANSEN_DST / f"{tile}_{layer}.tif"
        if dst_path.exists():
            print(f"  [skip] {dst_path}")
            continue
        with rasterio.open(src_path) as src:
            row_top, col_left = src.index(w, n)
            row_bot, col_right = src.index(e, s)
            window = rasterio.windows.Window(
                col_left, row_top, col_right - col_left, row_bot - row_top
            )
            arr = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                height=arr.shape[0], width=arr.shape[1],
                transform=transform,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256,
            )
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(arr, 1)
        print(f"  wrote {dst_path}  shape={arr.shape}")


def main():
    tiles = sys.argv[1:] or [t for t in all_tile_ids() if t.startswith("CMR_")]
    print(f"Cropping Hansen for {len(tiles)} tiles: {tiles}")
    for t in tiles:
        crop_one(t)


if __name__ == "__main__":
    main()
