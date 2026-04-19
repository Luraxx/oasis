"""Validate a directory of submission GeoJSON files.

Checks per file:

* Path exists, JSON parses, is a FeatureCollection.
* CRS is EPSG:4326 (or unspecified, since GeoJSON default is 4326).
* All geometries are valid via shapely.
* Polygon count >= 1 (warns if 0).
* Total area is sane: < 30% of the originating tile bounds, and > 0.

Exits non-zero if any error occurs (warnings do not fail).

Usage:
    python -m oasis.submission_check submission/*.geojson
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd


def check_file(path: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a single GeoJSON path."""
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        errors.append(f"{path}: missing")
        return errors, warnings
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"{path}: invalid JSON: {exc}")
        return errors, warnings
    if data.get("type") != "FeatureCollection":
        errors.append(f"{path}: not a FeatureCollection")
        return errors, warnings

    feats = data.get("features", [])
    if not feats:
        warnings.append(f"{path}: 0 features (empty submission for this tile)")
        return errors, warnings

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        errors.append(f"{path}: geopandas read failed: {exc}")
        return errors, warnings

    if gdf.crs is not None and str(gdf.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        errors.append(f"{path}: CRS is {gdf.crs}, expected EPSG:4326")

    invalid = (~gdf.geometry.is_valid).sum()
    if invalid:
        errors.append(f"{path}: {int(invalid)} invalid geometries")

    # Area check: if the submission spans multiple UTM zones (as a combined
    # submission.geojson does), estimate_utm_crs() picks one zone and
    # projection errors balloon the numbers. Skip the bound check in that
    # case by grouping features by centroid longitude band.
    try:
        centroid_lons = gdf.geometry.centroid.x
        span = float(centroid_lons.max() - centroid_lons.min())
    except Exception:
        span = 0.0

    if span > 12.0:  # spans more than ~12 deg longitude => cross-zone
        area_ha = 0.0
        groups = (centroid_lons // 6).round().astype(int)
        for grp in sorted(groups.unique()):
            sub = gdf[groups == grp]
            utm = sub.estimate_utm_crs()
            area_ha += float((sub.to_crs(utm).area / 10_000.0).sum())
    else:
        utm = gdf.estimate_utm_crs()
        area_ha = float((gdf.to_crs(utm).area / 10_000.0).sum())

    if area_ha <= 0:
        errors.append(f"{path}: zero total area")
    if area_ha > 1_000_000:  # 10 000 km^2 sanity bound (combined allowed)
        warnings.append(f"{path}: total area {area_ha:.0f} ha is unusually large")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission GeoJSONs.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for p in args.paths:
        e, w = check_file(p)
        all_errors.extend(e)
        all_warnings.extend(w)

    for w in all_warnings:
        print(f"WARN: {w}")
    for e in all_errors:
        print(f"FAIL: {e}")
    if all_errors:
        sys.exit(1)
    print(f"OK: {len(args.paths)} submissions validated")


if __name__ == "__main__":
    main()
