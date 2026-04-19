"""Build final submission variants:
1. sweep_opt: Directly from sweep results (max local IoU)
2. hybrid_safe: Toned-down version with light gating (hedge against overfitting to weak labels)
3. hybrid_mid: Middle ground
"""
from __future__ import annotations

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

# === VARIANT 1: SWEEP OPTIMIZED ===
SWEEP_OPT = {
    "18NVJ_1_6": {
        "w_ekl": 0.24, "w_luis": 0.16, "w_cmr": 0.60,
        "threshold": 0.40, "sigma": 0.5,
        "close": 1, "open": 1, "dilate": 0,
        "min_ha": 0.25, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
    "18NYH_2_1": {
        "w_ekl": 0.15, "w_luis": 0.10, "w_cmr": 0.75,
        "threshold": 0.40, "sigma": 0.5,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.25, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
    "33NTE_5_1": {
        "w_ekl": 0.24, "w_luis": 0.16, "w_cmr": 0.60,
        "threshold": 0.40, "sigma": 2.0,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
    "47QMA_6_2": {
        "w_ekl": 0.15, "w_luis": 0.10, "w_cmr": 0.75,
        "threshold": 0.20, "sigma": 2.0,
        "close": 2, "open": 0, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": True, "erode": 0,
        "hansen_min_ha": 0.20,
    },
    "48PWA_0_6": {
        "w_ekl": 0.15, "w_luis": 0.10, "w_cmr": 0.75,
        "threshold": 0.40, "sigma": 0.3,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
}

# === VARIANT 2: HYBRID SAFE (toned down - hedge against overfitting) ===
HYBRID_SAFE = {
    "18NVJ_1_6": {
        "w_ekl": 0.35, "w_luis": 0.25, "w_cmr": 0.40,
        "threshold": 0.35, "sigma": 0.5,
        "close": 1, "open": 1, "dilate": 0,
        "min_ha": 0.25, "final_min_ha": 0.15,
        "model_gate_thr": 0.10, "model_gate_radius": 3,
        "hansen_standalone": False, "erode": 0,
    },
    "18NYH_2_1": {
        "w_ekl": 0.30, "w_luis": 0.20, "w_cmr": 0.50,
        "threshold": 0.35, "sigma": 0.5,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.15,
        "model_gate_thr": 0.10, "model_gate_radius": 3,
        "hansen_standalone": False, "erode": 1,
    },
    "33NTE_5_1": {
        "w_ekl": 0.30, "w_luis": 0.20, "w_cmr": 0.50,
        "threshold": 0.35, "sigma": 1.0,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.15, "final_min_ha": 0.15,
        "model_gate_thr": 0.10, "model_gate_radius": 4,
        "hansen_standalone": False, "erode": 0,
    },
    "47QMA_6_2": {
        "w_ekl": 0.20, "w_luis": 0.10, "w_cmr": 0.70,
        "threshold": 0.20, "sigma": 1.5,
        "close": 2, "open": 0, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.15,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": True, "erode": 1,
        "hansen_min_ha": 0.30,
    },
    "48PWA_0_6": {
        "w_ekl": 0.25, "w_luis": 0.15, "w_cmr": 0.60,
        "threshold": 0.35, "sigma": 0.3,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.15, "final_min_ha": 0.15,
        "model_gate_thr": 0.08, "model_gate_radius": 4,
        "hansen_standalone": False, "erode": 0,
    },
}

# === VARIANT 3: HYBRID MID (between sweep and safe) ===
HYBRID_MID = {
    "18NVJ_1_6": {
        "w_ekl": 0.30, "w_luis": 0.20, "w_cmr": 0.50,
        "threshold": 0.38, "sigma": 0.5,
        "close": 1, "open": 1, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
    "18NYH_2_1": {
        "w_ekl": 0.20, "w_luis": 0.15, "w_cmr": 0.65,
        "threshold": 0.38, "sigma": 0.5,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
    "33NTE_5_1": {
        "w_ekl": 0.25, "w_luis": 0.15, "w_cmr": 0.60,
        "threshold": 0.38, "sigma": 1.5,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.15, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
    "47QMA_6_2": {
        "w_ekl": 0.15, "w_luis": 0.10, "w_cmr": 0.75,
        "threshold": 0.20, "sigma": 2.0,
        "close": 2, "open": 0, "dilate": 0,
        "min_ha": 0.20, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": True, "erode": 0,
        "hansen_min_ha": 0.20,
    },
    "48PWA_0_6": {
        "w_ekl": 0.20, "w_luis": 0.10, "w_cmr": 0.70,
        "threshold": 0.38, "sigma": 0.3,
        "close": 1, "open": 0, "dilate": 0,
        "min_ha": 0.15, "final_min_ha": 0.10,
        "model_gate_thr": 0.0, "model_gate_radius": 0,
        "hansen_standalone": False, "erode": 0,
    },
}

ALL_VARIANTS = {
    "final_sweep_opt": SWEEP_OPT,
    "final_hybrid_safe": HYBRID_SAFE,
    "final_hybrid_mid": HYBRID_MID,
}


def load_prob(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype(np.float32)
        prof = s.profile.copy()
        meta = {"height": s.height, "width": s.width,
                "transform": s.transform, "crs": s.crs}
    if a.max() > 10: a /= 1000.0
    return a, prof, meta

def reproj(p, meta, dtype=np.uint8):
    if not p.exists(): return None
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
    if raw is None: return np.zeros((meta["height"], meta["width"]), bool)
    days = raw % 10000; conf = raw // 10000
    return (conf >= 2) & (days > POST_CUTOFF_LOW) & (days <= POST_CUTOFF_HIGH)

def load_gladl_post2020(tile, meta, min_conf=2):
    out = np.zeros((meta["height"], meta["width"]), bool)
    for yy in [21,22,23,24,25]:
        p = LABEL_DIR / "gladl" / f"gladl_{tile}_alert{yy:02d}.tif"
        conf = reproj(p, meta, dtype=np.uint8)
        if conf is not None: out |= (conf >= min_conf)
    return out

def postprocess(binary, close, open_, dilate_, min_ha, erode_=0):
    res = binary.astype(np.uint8)
    if close > 0: res = nd.binary_closing(res, nd.generate_binary_structure(2,2), iterations=close).astype(np.uint8)
    if open_ > 0: res = nd.binary_opening(res, nd.generate_binary_structure(2,1), iterations=open_).astype(np.uint8)
    if dilate_ > 0: res = nd.binary_dilation(res, nd.generate_binary_structure(2,1), iterations=dilate_).astype(np.uint8)
    if erode_ > 0: res = nd.binary_erosion(res, nd.generate_binary_structure(2,1), iterations=erode_).astype(np.uint8)
    if min_ha > 0:
        min_px = int(min_ha * 100)
        labeled, n = nd.label(res)
        if n > 0:
            sizes = nd.sum(res, labeled, index=range(1, n+1))
            keep = np.zeros(n+2, dtype=bool); keep[1:n+1] = sizes >= min_px
            res = keep[labeled].astype(np.uint8)
    return res


def build_submission(name, tile_configs):
    out_dir = ROOT / "submissions" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Building {name} ===", flush=True)

    summary = {}
    all_features = []
    total_px = 0

    for tile in TEST_TILES:
        cfg = tile_configs[tile]
        ekl, prof, meta = load_prob(EKL_DIR / f"prob_{tile}.tif")
        luis, _, _ = load_prob(LUIS_DIR / f"prob_{tile}.tif")
        cmr, _, _ = load_prob(CMR_DIR / f"prob_{tile}.tif")

        shapes = [ekl.shape, luis.shape, cmr.shape]
        h, w = min(s[0] for s in shapes), min(s[1] for s in shapes)
        ekl, luis, cmr = ekl[:h,:w], luis[:h,:w], cmr[:h,:w]

        we, wl, wc = cfg["w_ekl"], cfg["w_luis"], cfg["w_cmr"]
        ws = we + wl + wc
        we, wl, wc = we/ws, wl/ws, wc/ws
        prob = we * ekl + wl * luis + wc * cmr

        if cfg.get("sigma", 0) > 0:
            prob = gaussian_filter(prob, sigma=cfg["sigma"])

        binary = (prob >= cfg["threshold"]).astype(np.uint8)
        binary = postprocess(binary, cfg["close"], cfg["open"],
                           cfg.get("dilate", 0), cfg["min_ha"]).astype(bool)

        # Weak label union
        ly = reproj(HANSEN_DIR / f"{tile}_lossyear.tif", meta)
        tc = reproj(HANSEN_DIR / f"{tile}_treecover2000.tif", meta)
        forest = tc[:h,:w] >= 30 if tc is not None else None

        gate_thr = cfg.get("model_gate_thr", 0)
        gate_radius = cfg.get("model_gate_radius", 0)
        model_near = nd.binary_dilation(prob >= gate_thr, iterations=gate_radius) if gate_thr > 0 and gate_radius > 0 else np.ones((h,w), bool)

        hansen_min = cfg.get("hansen_min_ha", 0.30)
        if ly is not None and forest is not None:
            ly_c = ly[:h,:w]
            hansen_pos = forest & (ly_c >= 21) & (ly_c <= 24)
            if cfg.get("hansen_standalone"):
                hansen_c = postprocess(hansen_pos, 0, 0, 0, hansen_min).astype(bool)
            else:
                hansen_c = postprocess(hansen_pos & model_near, 0, 0, 0, hansen_min).astype(bool)
            binary = binary | hansen_c

        radd_pos = load_radd_post2020(tile, meta)[:h,:w]
        if forest is not None: radd_pos = radd_pos & forest
        radd_c = postprocess(radd_pos & model_near, 0, 0, 0, 0.10).astype(bool)
        binary = binary | radd_c

        gladl_pos = load_gladl_post2020(tile, meta)[:h,:w]
        if forest is not None: gladl_pos = gladl_pos & forest
        gladl_c = postprocess(gladl_pos & model_near, 0, 0, 0, 0.10).astype(bool)
        binary = binary | gladl_c

        binary = postprocess(binary, 0, 0, 0, cfg["final_min_ha"],
                           erode_=cfg.get("erode", 0)).astype(bool)

        print(f"  [{tile}] -> {int(binary.sum()):,}px", flush=True)

        # Write
        out_bin = out_dir / f"pred_{tile}.tif"
        bm = prof.copy()
        bm.update(dtype="uint8", count=1, compress="LZW", nodata=0,
                  height=binary.shape[0], width=binary.shape[1])
        with rasterio.open(out_bin, "w", **bm) as dst:
            dst.write(binary.astype(np.uint8), 1)

        gj_path = out_dir / f"pred_{tile}.geojson"
        if not binary.any():
            gj_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
            summary[tile] = {"polys": 0, "binary_px": 0}
            all_features.append((tile, []))
            continue
        try:
            gj = raster_to_geojson(out_bin, output_path=gj_path, min_area_ha=cfg["final_min_ha"])
        except ValueError:
            gj = {"type": "FeatureCollection", "features": []}
            gj_path.write_text(json.dumps(gj))
            summary[tile] = {"polys": 0, "binary_px": 0}
            all_features.append((tile, []))
            continue

        # Add metadata
        comp_labels, n_comp = cc_label(binary)
        for i, feat in enumerate(gj["features"]):
            cid = i + 1
            if cid > n_comp: continue
            pmask = (comp_labels == cid)
            if not pmask.any(): continue
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

    cgj = {"type": "FeatureCollection",
           "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
           "features": [f for _, fts in all_features for f in fts]}
    (out_dir / "submission.geojson").write_text(json.dumps(cgj))
    (out_dir / "summary.json").write_text(json.dumps({
        "per_tile": summary,
        "total_polys": sum(v.get("polys", 0) for v in summary.values()),
        "total_px": total_px,
    }, indent=2))
    print(f"  TOTAL: polys={len(cgj['features'])} px={total_px:,}", flush=True)


for name, configs in ALL_VARIANTS.items():
    build_submission(name, configs)

print("\nAll done!", flush=True)
