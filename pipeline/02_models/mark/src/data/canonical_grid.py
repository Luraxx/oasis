"""Deterministic per-tile 10 m UTM grid derived from challenge metadata.

Every challenge tile is exactly 10 km × 10 km, axis-aligned in its local UTM
zone. The metadata GeoJSONs carry the upper-left UTM corner in the `origin`
field (EWKT ``SRID=<epsg>;POINT(ul_x ul_y)``). We use that corner, 10 m pixel
size, and a 1000×1000 shape as the canonical grid — independent of any source
raster's quirks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

ROOT = Path("/shared-docker/oasis-mark-2")
META_DIR = ROOT / "data/makeathon-challenge/metadata"
# Writable location for user-added tiles (data/ is root-owned, can't modify originals)
EXTRA_META_DIR = ROOT / "external/metadata"

# 10 km / 10 m → 1000 pixels on each side. Exact, no buffer.
PIXEL_SIZE_M = 10.0
TILE_EXTENT_M = 10_000.0
GRID_SHAPE = (int(TILE_EXTENT_M / PIXEL_SIZE_M), int(TILE_EXTENT_M / PIXEL_SIZE_M))  # (H, W) = (1000, 1000)

_ORIGIN_RE = re.compile(r"SRID=(\d+);POINT\(([-\d.]+)\s+([-\d.]+)\)")


@dataclass(frozen=True)
class CanonicalGrid:
    tile_id: str
    split: str  # "train" | "test"
    epsg: int
    ul_x: float
    ul_y: float
    shape: tuple[int, int] = GRID_SHAPE  # (H, W)
    pixel_size: float = PIXEL_SIZE_M

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    @property
    def transform(self) -> Affine:
        return Affine(
            self.pixel_size, 0.0, self.ul_x,
            0.0, -self.pixel_size, self.ul_y,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in the tile's UTM CRS."""
        h, w = self.shape
        return (
            self.ul_x,
            self.ul_y - h * self.pixel_size,
            self.ul_x + w * self.pixel_size,
            self.ul_y,
        )

    def rasterio_profile(self, count: int = 1, dtype: str = "uint8", nodata: float | None = 0) -> dict:
        """Return a rasterio profile dict with this grid, suitable for `rasterio.open(..., "w", **profile)`."""
        return {
            "driver": "GTiff",
            "height": self.shape[0],
            "width": self.shape[1],
            "count": count,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": nodata,
        }


def _parse_origin(origin_ewkt: str) -> tuple[int, float, float]:
    m = _ORIGIN_RE.match(origin_ewkt.strip())
    if not m:
        raise ValueError(f"Unrecognised origin EWKT: {origin_ewkt!r}")
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


@lru_cache(maxsize=1)
def _load_all() -> dict[str, CanonicalGrid]:
    out: dict[str, CanonicalGrid] = {}
    sources = [
        ("train", META_DIR / "train_tiles.geojson"),
        ("test", META_DIR / "test_tiles.geojson"),
        # Supplemental writable extras — e.g. Cameroon tiles we pulled ourselves.
        ("train", EXTRA_META_DIR / "train_tiles_extra.geojson"),
    ]
    for split, path in sources:
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        for _, row in gdf.iterrows():
            epsg, ul_x, ul_y = _parse_origin(row["origin"])
            out[row["name"]] = CanonicalGrid(
                tile_id=row["name"], split=split,
                epsg=epsg, ul_x=ul_x, ul_y=ul_y,
            )
    return out


def grid_for(tile_id: str) -> CanonicalGrid:
    grids = _load_all()
    if tile_id not in grids:
        raise KeyError(f"Unknown tile {tile_id!r}. Known: {sorted(grids)[:5]}...")
    return grids[tile_id]


def all_tile_ids(split: str | None = None) -> list[str]:
    grids = _load_all()
    if split is None:
        return sorted(grids)
    return sorted(tid for tid, g in grids.items() if g.split == split)


if __name__ == "__main__":
    # Verify canonical grid matches actual S2 rasters (should overlap ~100 %)
    grids = _load_all()
    for tile_id, g in sorted(grids.items()):
        s2_dir = ROOT / f"data/makeathon-challenge/sentinel-2/{g.split}/{tile_id}__s2_l2a"
        files = sorted(s2_dir.glob("*.tif"))
        # pick first with min(shape) >= 900
        actual = None
        for f in files:
            with rasterio.open(f) as src:
                if min(src.shape) >= 900 and str(src.crs) == f"EPSG:{g.epsg}":
                    actual = (src.bounds, src.shape, src.crs)
                    break
        if actual is None:
            print(f"  {tile_id} [{g.split}]: NO usable S2 to compare")
            continue
        ab, ashape, acrs = actual
        # intersection over canonical area
        gb = g.bounds
        ix = max(gb[0], ab.left); iy = max(gb[1], ab.bottom)
        ax = min(gb[2], ab.right); ay = min(gb[3], ab.top)
        inter = max(0.0, ax - ix) * max(0.0, ay - iy)
        canon_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
        actual_area = (ab.right - ab.left) * (ab.top - ab.bottom)
        iou = inter / (canon_area + actual_area - inter) if (canon_area + actual_area - inter) > 0 else 0.0
        print(f"  {tile_id} [{g.split}] epsg={g.epsg} shape={g.shape} | "
              f"S2 shape={ashape} iou={iou:.4f}")
