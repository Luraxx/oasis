"""Predict test submissions with a per-tile adaptive threshold.

Instead of a fixed threshold (which leaves 33NTE and 47QMA silent because their
per-pixel probs never exceed ~0.01), pick threshold so each tile outputs a
target fraction of its forest pixels as positive. Rationale: the model may be
systematically under-confident on unfamiliar biomes, but its *relative ranking*
of pixels within a tile is still informative. Quantile-based thresholding
extracts that signal.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import rasterio

ROOT = Path("/shared-docker/oasis-mark-2")
sys.path.insert(0, str(ROOT))

from src.data.canonical_grid import all_tile_ids, grid_for  # noqa: E402
from src.model.dataset import load_tile  # noqa: E402
from submission_utils import raster_to_geojson  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SUBMISSIONS = ROOT / "submissions"
SUBMISSIONS.mkdir(parents=True, exist_ok=True)


def predict_tile_adaptive(
    model: lgb.Booster, tile: str, pos_rate: float, min_area_ha: float,
    floor_thr: float = 1e-4, binary_raster_dir: Path | None = None,
) -> dict:
    """Predict and binarise at per-tile quantile threshold targeting `pos_rate`.

    Rule:
      * If top-pos_rate pixels all have prob ≤ floor_thr, don't predict (tile is truly quiet).
      * Otherwise pick threshold = pos_rate-th percentile FROM TOP of prob distribution.
    """
    grid = grid_for(tile)
    bundle = load_tile(tile, sample_neg_ratio=None)
    H, W = bundle.shape
    n_forest = bundle.X.shape[0]
    probs = model.predict(bundle.X)
    target_n = int(n_forest * pos_rate)

    if target_n == 0 or probs.max() < floor_thr:
        log.warning(f"[{tile}] quiet (max prob {probs.max():.4f}); empty stub")
        out_path = SUBMISSIONS / f"{tile}.geojson"
        out_path.write_text('{"type":"FeatureCollection","features":[]}')
        return {"tile": tile, "threshold": None, "n_pos": 0, "n_poly": 0, "geojson": str(out_path)}

    # Top N probs: threshold = (n_forest - target_n)-th order statistic
    sorted_probs = np.partition(probs, n_forest - target_n)
    thr = max(float(sorted_probs[n_forest - target_n]), floor_thr)
    bin_forest = probs > thr
    if not bin_forest.any():
        bin_forest = probs >= thr
    n_pos = int(bin_forest.sum())
    pos_frac = n_pos / n_forest
    log.info(f"[{tile}] target={pos_rate:.2%} thr={thr:.5f} "
             f"actual={pos_frac:.2%} ({n_pos:,}/{n_forest:,} forest px) "
             f"max_prob={probs.max():.3f}")

    # Reconstruct full tile raster
    binary = np.zeros(H * W, dtype=np.uint8)
    binary[bundle.pixel_index[bin_forest]] = 1
    binary = binary.reshape(H, W)

    profile = grid.rasterio_profile(count=1, dtype="uint8", nodata=0)
    if binary_raster_dir is not None:
        binary_raster_dir.mkdir(parents=True, exist_ok=True)
        bin_path = binary_raster_dir / f"{tile}_binary.tif"
    else:
        bin_path = Path(tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name)
    with rasterio.open(bin_path, "w", **profile) as dst:
        dst.write(binary, 1)

    out_path = SUBMISSIONS / f"{tile}.geojson"
    try:
        gj = raster_to_geojson(bin_path, output_path=out_path, min_area_ha=min_area_ha)
        n_poly = len(gj["features"])
    except ValueError as e:
        log.warning(f"[{tile}] raster_to_geojson refused: {e}")
        out_path.write_text('{"type":"FeatureCollection","features":[]}')
        n_poly = 0

    if binary_raster_dir is None:
        bin_path.unlink(missing_ok=True)
    return {"tile": tile, "threshold": thr, "n_pos": n_pos, "n_poly": n_poly,
            "pos_frac": pos_frac, "geojson": str(out_path)}


def combine_to_single_geojson(results: list[dict], out: Path):
    features = []
    for r in results:
        gj_path = Path(r["geojson"])
        if not gj_path.exists():
            continue
        fc = json.loads(gj_path.read_text())
        for feat in fc.get("features", []):
            props = feat.get("properties") or {}
            props["tile_id"] = r["tile"]
            feat["properties"] = props
            features.append(feat)
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
        "features": features,
    }
    out.write_text(json.dumps(payload))
    log.info(f"Combined → {out}  features={len(features)}  size={out.stat().st_size/1024:.1f} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm_final.txt")
    ap.add_argument("--pos-rate", type=float, default=0.08,
                    help="Target per-tile positive fraction (of forest pixels).")
    ap.add_argument("--min-area-ha", type=float, default=0.1)
    ap.add_argument("--tiles", nargs="+", default=None)
    ap.add_argument("--keep-rasters", action="store_true")
    ap.add_argument("--out", default="submission_adaptive.geojson",
                    help="Output combined GeoJSON filename under submissions/")
    args = ap.parse_args()

    model_path = ROOT / "models" / args.model
    log.info(f"Loading model: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))

    test_tiles = args.tiles or sorted([t for t in all_tile_ids() if grid_for(t).split == "test"])
    log.info(f"Predicting on {len(test_tiles)} tile(s): {test_tiles}")
    log.info(f"pos_rate={args.pos_rate}  min_area_ha={args.min_area_ha}")

    bin_dir = SUBMISSIONS / "binaries_adaptive" if args.keep_rasters else None
    t0 = time.time()
    results = []
    for tile in test_tiles:
        try:
            results.append(predict_tile_adaptive(
                booster, tile, args.pos_rate, args.min_area_ha, binary_raster_dir=bin_dir,
            ))
        except Exception as e:
            log.error(f"[{tile}] failed: {e!r}")

    log.info(f"Per-tile predictions done in {time.time() - t0:.1f}s")
    for r in results:
        log.info(f"  {r}")

    combine_to_single_geojson(results, SUBMISSIONS / args.out)


if __name__ == "__main__":
    main()
