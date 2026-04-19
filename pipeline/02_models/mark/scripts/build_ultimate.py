"""Ultimate submission builder: tile-specific ensemble weights + Gaussian smoothing
+ optimized per-tile thresholds + smart weak label fusion + boundary refinement.

Key insight from error analysis:
- 47QMA_6_2: ekl+luis are nearly blind (IoU 0.107), CMR has massive signal
- 33NTE_5_1: CMR adds 6.7K confirmed unique pixels
- 48PWA_0_6: CMR adds 12.8K confirmed unique pixels
- 18NYH_2_1: Over-predicting (56K FP), need higher threshold
- 18NVJ_1_6: Sparse, need conservative approach

Strategy: tile-specific ensemble weights giving CMR much more weight where it helps.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage as nd
from scipy.ndimage import label as cc_label, gaussian_filter

ROOT = Path("/shared-docker/oasis-mark-2")
sys.path.insert(0, str(ROOT))

from submission_utils import raster_to_geojson

EKL_DIR = Path("/shared-docker/oasis-luis-v5/ekl_submission")
LUIS_DIR = Path("/shared-docker/oasis-luis-v5/luis_v4_submission")
CMR_DIR = ROOT / "external/cmr_prob"
HANSEN_DIR = ROOT / "external/hansen/cropped"
LABEL_DIR = ROOT / "external/makeathon-extras/labels/train"

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]
RADD_EPOCH = date(2014, 12, 31)
POST_CUTOFF_LOW = (date(2020, 12, 31) - RADD_EPOCH).days
POST_CUTOFF_HIGH = (date(2025, 12, 31) - RADD_EPOCH).days


# === TILE-SPECIFIC CONFIGS ===
# Based on error analysis: per-tile weights and thresholds
TILE_CONFIGS = {
    "18NVJ_1_6": {
        # Sparse forest. ekl is best, CMR adds a little.
        # IoU vs labels=0.175, lots of FP. Be conservative.
        "w_ekl": 0.55, "w_luis": 0.35, "w_cmr": 0.10,
        "threshold": 0.22,
        "close": 1, "open": 1, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.25,
        "sigma": 0.8,  # light smoothing
        "hansen_union": True, "radd_union": True, "gladl_union": True,
        "model_gate_thr": 0.12,  # gate weak labels
        "model_gate_radius": 3,
        "erode": 1,  # boundary tightening
    },
    "18NYH_2_1": {
        # Dense tropical. Models agree well. Main issue: 56K FP.
        # Need higher threshold + FP control.
        "w_ekl": 0.55, "w_luis": 0.35, "w_cmr": 0.10,
        "threshold": 0.25,
        "close": 2, "open": 0, "dilate": 1,
        "min_ha": 0.15, "final_min_ha": 0.20,
        "sigma": 1.0,  # moderate smoothing to clean boundaries
        "hansen_union": True, "radd_union": True, "gladl_union": True,
        "model_gate_thr": 0.15,
        "model_gate_radius": 3,
        "erode": 2,  # tighten boundaries to reduce FP
    },
    "33NTE_5_1": {
        # Mixed terrain. CMR adds 6.7K confirmed pixels.
        # Balanced errors (24K FP, 24K FN).
        "w_ekl": 0.45, "w_luis": 0.30, "w_cmr": 0.25,
        "threshold": 0.22,
        "close": 2, "open": 0, "dilate": 1,
        "min_ha": 0.10, "final_min_ha": 0.15,
        "sigma": 0.8,
        "hansen_union": True, "radd_union": True, "gladl_union": True,
        "model_gate_thr": 0.10,
        "model_gate_radius": 4,
        "erode": 1,
    },
    "47QMA_6_2": {
        # CRITICAL: ekl+luis nearly blind (IoU 0.107). CMR has 42K unique pixels.
        # Hansen has 53K pixels. This tile needs radical treatment.
        "w_ekl": 0.20, "w_luis": 0.10, "w_cmr": 0.70,
        "threshold": 0.15,  # very low — CMR is our main signal
        "close": 2, "open": 0, "dilate": 2,
        "min_ha": 0.10, "final_min_ha": 0.15,
        "sigma": 1.2,  # smooth CMR noise
        "hansen_union": True, "radd_union": False, "gladl_union": True,
        "model_gate_thr": 0.0,  # don't gate — model is too weak here
        "model_gate_radius": 0,
        "hansen_standalone": True,  # Add Hansen directly without gating
        "erode": 1,
    },
    "48PWA_0_6": {
        # Africa. CMR adds 12.8K confirmed. Balanced errors.
        "w_ekl": 0.40, "w_luis": 0.30, "w_cmr": 0.30,
        "threshold": 0.22,
        "close": 2, "open": 0, "dilate": 1,
        "min_ha": 0.10, "final_min_ha": 0.15,
        "sigma": 0.8,
        "hansen_union": True, "radd_union": False, "gladl_union": True,
        "model_gate_thr": 0.08,
        "model_gate_radius": 4,
        "erode": 1,
    },
}


def load_prob(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype(np.float32)
        prof = s.profile.copy()
        meta = {"height": s.height, "width": s.width,
                "transform": s.transform, "crs": s.crs}
    if a.max() > 10:
        a /= 1000.0
    return a, prof, meta


def reproj(p, meta, dtype=np.uint8):
    if not p.exists():
        return None
    dst = np.zeros((meta["height"], meta["width"]), dtype=dtype)
    with rasterio.open(p) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=meta["transform"], dst_crs=meta["crs"],
                  resampling=Resampling.nearest)
    return dst


def load_radd_post2020(tile, meta):
    p = LABEL_DIR / "radd" / f"radd_{tile}_labels.tif"
    raw = reproj(p, meta, dtype=np.int32)
    if raw is None:
        return np.zeros((meta["height"], meta["width"]), bool)
    days = raw % 10000
    conf = raw // 10000
    return (conf >= 2) & (days > POST_CUTOFF_LOW) & (days <= POST_CUTOFF_HIGH)


def load_gladl_post2020(tile, meta, min_conf=2):
    out = np.zeros((meta["height"], meta["width"]), bool)
    for yy in [21, 22, 23, 24, 25]:
        p = LABEL_DIR / "gladl" / f"gladl_{tile}_alert{yy:02d}.tif"
        conf = reproj(p, meta, dtype=np.uint8)
        if conf is not None:
            out |= (conf >= min_conf)
    return out


def postprocess(binary, close, open_, dilate_, min_ha, erode_=0):
    res = binary.astype(np.uint8)
    if close > 0:
        res = nd.binary_closing(res, nd.generate_binary_structure(2, 2),
                                iterations=close).astype(np.uint8)
    if open_ > 0:
        res = nd.binary_opening(res, nd.generate_binary_structure(2, 1),
                                iterations=open_).astype(np.uint8)
    if dilate_ > 0:
        res = nd.binary_dilation(res, nd.generate_binary_structure(2, 1),
                                 iterations=dilate_).astype(np.uint8)
    if erode_ > 0:
        res = nd.binary_erosion(res, nd.generate_binary_structure(2, 1),
                                iterations=erode_).astype(np.uint8)
    if min_ha > 0:
        min_px = int(min_ha * 100)
        labeled, n = nd.label(res)
        if n > 0:
            sizes = nd.sum(res, labeled, index=range(1, n + 1))
            keep = np.zeros(n + 2, dtype=bool)
            keep[1:n + 1] = sizes >= min_px
            res = keep[labeled].astype(np.uint8)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="default",
                    help="Config variant to use")
    args = ap.parse_args()

    out_dir = ROOT / "submissions" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.name} | TILE-SPECIFIC ULTIMATE BUILD ===")

    summary = {}
    all_features = []
    total_px = 0

    for tile in TEST_TILES:
        cfg = TILE_CONFIGS[tile]
        print(f"\n--- {tile} ---")
        print(f"  weights: ekl={cfg['w_ekl']} luis={cfg['w_luis']} cmr={cfg['w_cmr']}")
        print(f"  threshold={cfg['threshold']} sigma={cfg['sigma']} erode={cfg['erode']}")

        # Load probability maps
        ekl_p = EKL_DIR / f"prob_{tile}.tif"
        luis_p = LUIS_DIR / f"prob_{tile}.tif"
        cmr_p = CMR_DIR / f"prob_{tile}.tif"

        ekl, prof, meta = load_prob(ekl_p)
        luis_arr, cmr_arr = None, None
        if luis_p.exists():
            luis_arr, _, _ = load_prob(luis_p)
        if cmr_p.exists():
            cmr_arr, _, _ = load_prob(cmr_p)

        # Crop to common size
        shapes = [ekl.shape]
        if luis_arr is not None: shapes.append(luis_arr.shape)
        if cmr_arr is not None: shapes.append(cmr_arr.shape)
        h, w = min(s[0] for s in shapes), min(s[1] for s in shapes)
        ekl = ekl[:h, :w]
        if luis_arr is not None: luis_arr = luis_arr[:h, :w]
        if cmr_arr is not None: cmr_arr = cmr_arr[:h, :w]

        # Tile-specific weighted average
        we, wl, wc = cfg["w_ekl"], cfg["w_luis"], cfg["w_cmr"]
        if luis_arr is None: wl = 0
        if cmr_arr is None: wc = 0
        ws = we + wl + wc
        we, wl, wc = we/ws, wl/ws, wc/ws
        prob = we * ekl
        if luis_arr is not None: prob += wl * luis_arr
        if cmr_arr is not None: prob += wc * cmr_arr

        # Gaussian smoothing for cleaner boundaries
        if cfg["sigma"] > 0:
            prob = gaussian_filter(prob, sigma=cfg["sigma"])

        # Threshold
        thr = cfg["threshold"]
        binary = (prob >= thr).astype(np.uint8)
        binary = postprocess(binary, cfg["close"], cfg["open"],
                           cfg.get("dilate", 0), cfg["min_ha"]).astype(bool)

        print(f"  model_px={int(binary.sum()):,}")

        # Load external data
        ly = reproj(HANSEN_DIR / f"{tile}_lossyear.tif", meta)
        tc = reproj(HANSEN_DIR / f"{tile}_treecover2000.tif", meta)
        forest = None
        if tc is not None:
            forest = tc[:h, :w] >= 30

        used = [f"model({int(binary.sum())})"]

        # Model-near zone for gating weak labels
        gate_thr = cfg["model_gate_thr"]
        gate_radius = cfg["model_gate_radius"]
        if gate_thr > 0 and gate_radius > 0:
            model_near = nd.binary_dilation(prob >= gate_thr,
                                            iterations=gate_radius)
        else:
            model_near = np.ones((h, w), bool)  # no gating

        # Hansen union
        if cfg["hansen_union"] and ly is not None and forest is not None:
            ly_c = ly[:h, :w]
            hansen_pos = forest & (ly_c >= 21) & (ly_c <= 24)
            if cfg.get("hansen_standalone"):
                # Don't gate by model — add Hansen directly
                hansen_clean = postprocess(hansen_pos, 0, 0, 0, 0.30).astype(bool)
            else:
                hansen_gated = hansen_pos & model_near
                hansen_clean = postprocess(hansen_gated, 0, 0, 0, 0.30).astype(bool)
            if hansen_clean.any():
                binary = binary | hansen_clean
                used.append(f"hansen({int(hansen_clean.sum())})")

        # RADD union
        if cfg["radd_union"]:
            radd_pos = load_radd_post2020(tile, meta)[:h, :w]
            if forest is not None:
                radd_pos = radd_pos & forest
            radd_gated = radd_pos & model_near
            radd_clean = postprocess(radd_gated, 0, 0, 0, 0.10).astype(bool)
            if radd_clean.any():
                binary = binary | radd_clean
                used.append(f"radd({int(radd_clean.sum())})")

        # GLAD-L union
        if cfg["gladl_union"]:
            gladl_pos = load_gladl_post2020(tile, meta)[:h, :w]
            if forest is not None:
                gladl_pos = gladl_pos & forest
            gladl_gated = gladl_pos & model_near
            gladl_clean = postprocess(gladl_gated, 0, 0, 0, 0.10).astype(bool)
            if gladl_clean.any():
                binary = binary | gladl_clean
                used.append(f"gladl({int(gladl_clean.sum())})")

        # Final postprocess with boundary erosion
        binary = postprocess(binary, 0, 0, 0, cfg["final_min_ha"],
                           erode_=cfg.get("erode", 0)).astype(bool)

        print(f"  {used} -> final_px={int(binary.sum()):,}")

        # Write raster
        out_bin = out_dir / f"pred_{tile}.tif"
        bm = prof.copy()
        bm.update(dtype="uint8", count=1, compress="LZW", nodata=0,
                  height=binary.shape[0], width=binary.shape[1])
        with rasterio.open(out_bin, "w", **bm) as dst:
            dst.write(binary.astype(np.uint8), 1)

        # Vectorize
        gj_path = out_dir / f"pred_{tile}.geojson"
        if not binary.any():
            gj_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
            summary[tile] = {"polys": 0, "binary_px": 0}
            all_features.append((tile, []))
            continue
        try:
            gj = raster_to_geojson(out_bin, output_path=gj_path,
                                   min_area_ha=cfg["final_min_ha"])
        except ValueError:
            gj = {"type": "FeatureCollection", "features": []}
            gj_path.write_text(json.dumps(gj))
            summary[tile] = {"polys": 0, "binary_px": 0}
            all_features.append((tile, []))
            continue

        # Add metadata to features
        comp_labels, n_comp = cc_label(binary)
        for i, feat in enumerate(gj["features"]):
            cid = i + 1
            if cid > n_comp:
                continue
            pmask = (comp_labels == cid)
            if not pmask.any():
                continue
            conf = float(prob[pmask].mean())
            ts = None
            if ly is not None:
                ly_c = ly[:pmask.shape[0], :pmask.shape[1]]
                vals = ly_c[pmask]
                valid = vals[(vals >= 21) & (vals <= 24)]
                if valid.size > 0:
                    counts = np.bincount(valid, minlength=25)
                    yy = int(counts[21:25].argmax()) + 21
                    ts = (2000 + yy) * 100 + 6
            feat["properties"] = feat.get("properties") or {}
            feat["properties"]["confidence"] = round(conf, 3)
            feat["properties"]["time_step"] = int(ts) if ts is not None else None
            feat["properties"]["tile_id"] = tile

        gj_path.write_text(json.dumps(gj))
        summary[tile] = {"polys": len(gj["features"]), "binary_px": int(binary.sum())}
        total_px += int(binary.sum())
        all_features.append((tile, gj["features"]))

    # Combine all tiles
    cgj = {"type": "FeatureCollection",
           "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
           "features": [f for _, fts in all_features for f in fts]}
    (out_dir / "submission.geojson").write_text(json.dumps(cgj))
    (out_dir / "summary.json").write_text(json.dumps({
        "tile_configs": {t: {k: v for k, v in c.items()} for t, c in TILE_CONFIGS.items()},
        "per_tile": summary,
        "total_polys": sum(v.get("polys", 0) for v in summary.values()),
        "total_px": total_px,
    }, indent=2))
    print(f"\n=== COMBINED: polys={len(cgj['features'])}  total_px={total_px:,} ===")


if __name__ == "__main__":
    main()
