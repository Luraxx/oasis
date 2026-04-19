"""Fetch and cache external auxiliary layers.

Layers we want, in priority order of expected lift on deforestation:

1. ESA WorldCover 2021 v200    - 10 m global land cover (11 classes).
2. JRC Tropical Moist Forest   - annual undisturbed/degraded/regrowth.
3. Hansen GFC v1.11            - tree cover 2000 + lossyear 2001-2023.
4. OSM derived distance-to-road raster.

This module deliberately treats network access as optional. If the
fetch backend is unavailable (no internet, missing rasterio HTTP
support), every layer is logged and skipped, and the feature pack
silently falls back to zeros. That keeps the training pipeline working
in any environment while still benefiting from externals when present.

Usage:
    python -m oasis.externals fetch --split both
    python -m oasis.externals status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from oasis import audit, cache, paths


# Public COG endpoints. These can be substituted with locally mirrored
# COGs if network access is restricted.
WORLDCOVER_URL_TEMPLATE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
HANSEN_TREECOVER_URL = (
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/"
    "Hansen_GFC-2023-v1.11_treecover2000_{lat}_{lon}.tif"
)
HANSEN_LOSSYEAR_URL = (
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/"
    "Hansen_GFC-2023-v1.11_lossyear_{lat}_{lon}.tif"
)
# JRC TMF and OSM road distance are heavier - these endpoints are
# placeholders; users with the data locally can drop them under
# data/externals_raw/<layer>/. The fetcher falls through gracefully.
JRC_TMF_LOCAL_DIR = paths.EXTERNALS_ROOT / "_raw" / "jrc_tmf"
OSM_ROAD_LOCAL = paths.EXTERNALS_ROOT / "_raw" / "osm_road_distance.tif"


# WorldCover code -> dense index 0..10
WORLDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


def _hansen_tile_name(lat: float, lon: float) -> tuple[str, str]:
    """Hansen tiles are 10x10 deg, named by NW corner.

    e.g. for tile centred at (-3.5, -55) we want lat="00N" lon="060W" -> the NW
    corner is the upper-left of a 10-deg block.
    """
    lat_block = int(np.ceil(lat / 10.0)) * 10
    lon_block = int(np.floor(lon / 10.0)) * 10
    lat_str = f"{abs(lat_block):02d}{'N' if lat_block >= 0 else 'S'}"
    lon_str = f"{abs(lon_block):03d}{'E' if lon_block >= 0 else 'W'}"
    return lat_str, lon_str


def _worldcover_tile_name(lat: float, lon: float) -> str:
    """WorldCover tiles are 3x3 deg, NW-corner naming N06W081 etc."""
    lat_block = int(np.floor(lat / 3.0)) * 3
    lon_block = int(np.floor(lon / 3.0)) * 3
    lat_str = f"{'N' if lat_block >= 0 else 'S'}{abs(lat_block):02d}"
    lon_str = f"{'E' if lon_block >= 0 else 'W'}{abs(lon_block):03d}"
    return f"{lat_str}{lon_str}"


def _bounds_lonlat(meta: dict, arrays_meta: dict) -> tuple[float, float, float, float]:
    """Return tile bounds in EPSG:4326 (minlon, minlat, maxlon, maxlat)."""
    rmeta = cache.restore_rasterio_meta({"rasterio_meta": meta["rasterio_meta"]})
    transform = rmeta["transform"]
    crs = rmeta["crs"]
    h, w = arrays_meta["s2_stack"].shape[-2:]
    from rasterio.warp import transform_bounds

    left, bottom = transform * (0, h)
    right, top = transform * (w, 0)
    return transform_bounds(crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)


def _reproject_url_to_grid(
    url: str, dst_crs, dst_transform, dst_shape, resampling=Resampling.nearest
) -> np.ndarray | None:
    """Open a remote COG via rasterio /vsicurl and reproject. Returns None on failure."""
    try:
        with rasterio.Env(GDAL_HTTP_MULTIPLEX="YES", VSI_CACHE="YES"):
            with rasterio.open(f"/vsicurl/{url}") as src:
                src_crs = src.crs
                src_transform = src.transform
                src_data = src.read(1)
        out = np.zeros(dst_shape, dtype=src_data.dtype)
        reproject(
            source=src_data,
            destination=out,
            src_crs=src_crs,
            src_transform=src_transform,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            resampling=resampling,
        )
        return out
    except Exception as exc:
        print(f"[externals] fetch failed for {url}: {exc}")
        return None


def _save(out_dir: Path, name: str, arr: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.npy"
    np.save(p, arr)
    return p


def fetch_worldcover(
    tile_id: str, split: str, dst_crs, dst_transform, dst_shape, lonlat: tuple[float, float, float, float]
) -> np.ndarray | None:
    minlon, minlat, maxlon, maxlat = lonlat
    # WorldCover tiles cover 3x3 deg; our tiles are <=10 km so one tile suffices.
    tile_name = _worldcover_tile_name(minlat, minlon)
    url = WORLDCOVER_URL_TEMPLATE.format(tile=tile_name)
    raw = _reproject_url_to_grid(url, dst_crs, dst_transform, dst_shape, Resampling.nearest)
    if raw is None:
        return None

    dense = np.full(dst_shape, fill_value=255, dtype=np.uint8)
    for idx, code in enumerate(WORLDCOVER_CODES):
        dense[raw == code] = idx
    return dense


def fetch_hansen(
    tile_id: str, split: str, dst_crs, dst_transform, dst_shape, lonlat: tuple[float, float, float, float]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    minlon, minlat, maxlon, maxlat = lonlat
    cy = (minlat + maxlat) / 2
    cx = (minlon + maxlon) / 2
    lat_str, lon_str = _hansen_tile_name(cy, cx)
    tc_url = HANSEN_TREECOVER_URL.format(lat=lat_str, lon=lon_str)
    ly_url = HANSEN_LOSSYEAR_URL.format(lat=lat_str, lon=lon_str)
    tc = _reproject_url_to_grid(tc_url, dst_crs, dst_transform, dst_shape, Resampling.bilinear)
    ly = _reproject_url_to_grid(ly_url, dst_crs, dst_transform, dst_shape, Resampling.nearest)
    return tc, ly


def fetch_local_layer(
    src_path: Path, dst_crs, dst_transform, dst_shape, resampling=Resampling.nearest
) -> np.ndarray | None:
    if not src_path.exists():
        return None
    try:
        with rasterio.open(src_path) as src:
            data = src.read(1)
            src_crs = src.crs
            src_transform = src.transform
        out = np.zeros(dst_shape, dtype=data.dtype)
        reproject(
            source=data,
            destination=out,
            src_crs=src_crs,
            src_transform=src_transform,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            resampling=resampling,
        )
        return out
    except Exception as exc:
        print(f"[externals] failed reading {src_path}: {exc}")
        return None


def fetch_for_tile(tile_id: str, split: str, *, force: bool = False) -> dict[str, Path | None]:
    """Fetch every external layer for one tile and persist to disk."""
    arrays, meta = cache.load_tile_cache(tile_id, split)
    rmeta = cache.restore_rasterio_meta(meta)
    dst_crs = rmeta["crs"]
    dst_transform = rmeta["transform"]
    dst_shape = tuple(meta["shape"])

    out_dir = paths.EXTERNALS_ROOT / split / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lonlat = _bounds_lonlat(meta, arrays)

    written: dict[str, Path | None] = {}

    if force or not (out_dir / "worldcover.npy").exists():
        wc = fetch_worldcover(tile_id, split, dst_crs, dst_transform, dst_shape, lonlat)
        written["worldcover"] = _save(out_dir, "worldcover", wc) if wc is not None else None
    else:
        written["worldcover"] = out_dir / "worldcover.npy"

    if force or not (out_dir / "hansen_treecover2000.npy").exists():
        tc, ly = fetch_hansen(tile_id, split, dst_crs, dst_transform, dst_shape, lonlat)
        written["hansen_treecover2000"] = (
            _save(out_dir, "hansen_treecover2000", tc) if tc is not None else None
        )
        written["hansen_lossyear"] = (
            _save(out_dir, "hansen_lossyear", ly) if ly is not None else None
        )

    # JRC TMF (local only)
    for year in (2020, 2024):
        name = f"jrc_tmf_{year}"
        target = out_dir / f"{name}.npy"
        if not force and target.exists():
            written[name] = target
            continue
        candidate = JRC_TMF_LOCAL_DIR / f"jrc_tmf_{year}.tif"
        arr = fetch_local_layer(candidate, dst_crs, dst_transform, dst_shape, Resampling.nearest)
        written[name] = _save(out_dir, name, arr) if arr is not None else None

    # OSM road distance (local only)
    target = out_dir / "osm_road_distance_km.npy"
    if force or not target.exists():
        arr = fetch_local_layer(OSM_ROAD_LOCAL, dst_crs, dst_transform, dst_shape, Resampling.bilinear)
        written["osm_road_distance_km"] = _save(out_dir, "osm_road_distance_km", arr) if arr is not None else None
    else:
        written["osm_road_distance_km"] = target

    return written


def cmd_fetch(args: argparse.Namespace) -> None:
    paths.ensure_dirs()
    train_tiles, test_tiles = audit.audit(strict=False)
    todo: list[tuple[str, str]] = []
    if args.split in ("train", "both"):
        todo.extend((t, "train") for t in train_tiles)
    if args.split in ("test", "both"):
        todo.extend((t, "test") for t in test_tiles)

    for tile, split in todo:
        if not cache.cache_exists(tile, split):
            print(f"[externals] cache missing for {tile} ({split}); build cache first")
            continue
        try:
            written = fetch_for_tile(tile, split, force=args.force)
            ok = sum(1 for v in written.values() if v is not None)
            print(f"[externals] {tile} ({split}): {ok}/{len(written)} layers ready")
        except Exception as exc:
            print(f"[externals][error] {tile} ({split}): {exc}")


def cmd_status(_: argparse.Namespace) -> None:
    print(json.dumps({
        "externals_root": str(paths.EXTERNALS_ROOT),
        "jrc_tmf_local": str(JRC_TMF_LOCAL_DIR),
        "osm_road_local": str(OSM_ROAD_LOCAL),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="External auxiliary layers fetcher.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fetch = sub.add_parser("fetch", help="Download / reproject layers per tile.")
    p_fetch.add_argument("--split", choices=["train", "test", "both"], default="both")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_fetch)
    p_status = sub.add_parser("status")
    p_status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
