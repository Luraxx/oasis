#!/usr/bin/env python3
"""
Generate submission variants that differ ONLY in the Africa tile (33NTE_5_1).
All other tiles (18NVJ, 18NYH, 47QMA, 48PWA) remain exactly as v5_final.
"""
import sys, warnings, json, shutil, numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes as rio_shapes, rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
from scipy import ndimage
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import SUBMISSION

TILE = '33NTE_5_1'

# ──────────────── Load data ────────────────

def load_data():
    """Load all prediction sources for Africa tile."""
    # v5 prob (60% Ekl + 40% Luis, already fused)
    with rasterio.open(f'submission/v5_final/prob_{TILE}.tif') as src:
        v5_prob = src.read(1).astype(np.float32) / 1000.0
        prof = src.profile.copy()
    
    # Eklavya prob
    with rasterio.open(f'../oasis-eklavya/submission/prob_{TILE}.tif') as src:
        ekl_prob = src.read(1).astype(np.float32)
        if ekl_prob.max() > 10: ekl_prob /= 1000.0
    
    # Luis prob
    with rasterio.open(f'luis_v4_submission/prob_{TILE}.tif') as src:
        luis_prob = src.read(1).astype(np.float32)
        if luis_prob.max() > 10: luis_prob /= 1000.0
    
    # Hansen GFC lossyear
    with rasterio.open(f'../oasis-mark-2/external/hansen/cropped/{TILE}_lossyear.tif') as src:
        hansen = src.read(1)
        hprof = src.profile.copy()
    hansen_reproj = np.zeros_like(v5_prob, dtype=np.uint8)
    reproject(source=hansen, destination=hansen_reproj,
              src_transform=hprof['transform'], src_crs=hprof['crs'],
              dst_transform=prof['transform'], dst_crs=prof['crs'],
              resampling=Resampling.nearest)
    recent_loss = (hansen_reproj >= 20).astype(np.uint8)
    
    # Hansen treecover
    with rasterio.open(f'../oasis-mark-2/external/hansen/cropped/{TILE}_treecover2000.tif') as src:
        tc2000 = src.read(1)
        tc_prof = src.profile.copy()
    tc_reproj = np.zeros_like(v5_prob, dtype=np.uint8)
    reproject(source=tc2000, destination=tc_reproj,
              src_transform=tc_prof['transform'], src_crs=tc_prof['crs'],
              dst_transform=prof['transform'], dst_crs=prof['crs'],
              resampling=Resampling.nearest)
    
    # Mark-2 predictions
    mk2 = gpd.read_file(f'../oasis-mark-2/submissions/{TILE}.geojson')
    mk2_utm = mk2.to_crs(prof['crs'])
    mk2_mask = rasterize(mk2_utm.geometry, out_shape=v5_prob.shape,
                         transform=prof['transform'], fill=0, default_value=1, dtype=np.uint8)
    
    # Mark-1 predictions
    mk1 = gpd.read_file(f'../oasis-mark/submission/pred_{TILE}.geojson')
    mk1_utm = mk1.to_crs(prof['crs'])
    mk1_mask = rasterize(mk1_utm.geometry, out_shape=v5_prob.shape,
                         transform=prof['transform'], fill=0, default_value=1, dtype=np.uint8)
    
    return {
        'v5_prob': v5_prob, 'ekl_prob': ekl_prob, 'luis_prob': luis_prob,
        'prof': prof, 'recent_loss': recent_loss, 'treecover': tc_reproj,
        'mk2_mask': mk2_mask, 'mk1_mask': mk1_mask,
    }


def pp(binary, close=1, opn=2, min_px=50):
    r = binary.copy()
    if close > 0:
        r = ndimage.binary_closing(r, ndimage.generate_binary_structure(2, 2),
                                   iterations=close).astype(np.uint8)
    if opn > 0:
        r = ndimage.binary_opening(r, ndimage.generate_binary_structure(2, 1),
                                   iterations=opn).astype(np.uint8)
    if min_px > 1:
        lab, n = ndimage.label(r)
        sizes = ndimage.sum(r, lab, range(1, n + 1))
        for c in range(1, n + 1):
            if sizes[c - 1] < min_px:
                r[lab == c] = 0
    return r


def estimate_year(binary, prof, split='test'):
    """Year estimation using NBR drop."""
    h, w = binary.shape
    s2_dir = Path(f'data/sentinel-2/{split}/{TILE}__s2_l2a')
    
    yearly_nbr = {}
    for year in range(2020, 2026):
        nbrs = []
        for month in range(1, 13):
            p = s2_dir / f'{TILE}__s2_l2a_{year}_{month}.tif'
            if not p.exists(): continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                b08, b12 = b[7], b[11]
                d = b08 + b12
                nbr = np.where(d > 0, (b08 - b12) / d, 0)
                nbr[b[0] <= 0] = np.nan
                nbrs.append(nbr[:h, :w])
            except:
                continue
        if nbrs:
            with np.errstate(all='ignore'):
                yearly_nbr[year] = np.nanmedian(nbrs, axis=0)
    
    sorted_yrs = sorted(yr for yr in yearly_nbr.keys() if yr <= 2025)
    max_drop = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)
    
    for i in range(1, len(sorted_yrs)):
        prev, curr = sorted_yrs[i-1], sorted_yrs[i]
        p_arr = yearly_nbr[prev][:h, :w]
        c_arr = yearly_nbr[curr][:h, :w]
        mh = min(p_arr.shape[0], c_arr.shape[0], h)
        mw = min(p_arr.shape[1], c_arr.shape[1], w)
        with np.errstate(all='ignore'):
            drop = np.nan_to_num(p_arr[:mh, :mw] - c_arr[:mh, :mw], nan=0)
        drop = np.clip(drop, 0, None)
        if mh < h or mw < w:
            full = np.zeros((h, w), dtype=np.float32)
            full[:mh, :mw] = drop
            drop = full
        better = drop > max_drop
        max_drop[better] = drop[better]
        drop_year[better] = curr
    
    labeled, n = ndimage.label(binary)
    year_map = np.full((h, w), 2306, dtype=np.int32)
    for cid in range(1, n + 1):
        cmask = labeled == cid
        cyears = drop_year[cmask]
        cscores = max_drop[cmask]
        if cscores.sum() > 0:
            yt = {}
            for yr, sc in zip(cyears, cscores):
                yt[yr] = yt.get(yr, 0) + sc
            modal_yr = max(yt, key=yt.get)
        else:
            vals, cnts = np.unique(cyears, return_counts=True)
            modal_yr = vals[np.argmax(cnts)]
        modal_yr = max(2020, min(2025, modal_yr))
        year_map[cmask] = (modal_yr % 100) * 100 + 6
    return year_map


def mask_to_geojson(mask, fused_prob, prof, min_ha=0.5):
    """Convert binary mask to GeoDataFrame with time_step and confidence."""
    h, w = mask.shape
    if mask.sum() == 0:
        return gpd.GeoDataFrame(columns=['time_step', 'confidence', 'tile_id', 'geometry'],
                                geometry='geometry', crs='EPSG:4326')
    
    polys = [shape(geom) for geom, v in rio_shapes(
        mask.astype(np.uint8), mask=mask.astype(bool),
        transform=prof['transform']) if v == 1]
    if not polys:
        return gpd.GeoDataFrame(columns=['time_step', 'confidence', 'tile_id', 'geometry'],
                                geometry='geometry', crs='EPSG:4326')
    
    gdf = gpd.GeoDataFrame(geometry=polys, crs=prof['crs'])
    gdf = gdf.to_crs("EPSG:4326")
    utm = gdf.estimate_utm_crs()
    areas = gdf.to_crs(utm).area / 10000
    gdf = gdf[areas >= min_ha].reset_index(drop=True)
    if gdf.empty:
        return gdf
    
    # Year estimation
    year_map = estimate_year(mask, prof)
    
    # Assign time_step and confidence
    gdf_px = gdf.to_crs(prof['crs'])
    transform = prof['transform']
    ts_list, conf_list = [], []
    for _, row in gdf_px.iterrows():
        c = row.geometry.centroid
        col = int((c.x - transform.c) / transform.a)
        rpx = int((c.y - transform.f) / transform.e)
        ts, conf = 2306, 0.5
        if 0 <= rpx < h and 0 <= col < w:
            ts = int(year_map[rpx, col])
            conf = float(fused_prob[rpx, col])
        ts_list.append(ts)
        conf_list.append(round(min(max(conf, 0.01), 1.0), 3))
    
    gdf['time_step'] = ts_list
    gdf['confidence'] = conf_list
    gdf['tile_id'] = TILE
    return gdf


def build_submission(africa_gdf, variant_name):
    """Replace Africa tile in v5_final submission with new predictions."""
    # Load v5_final submission
    v5 = gpd.read_file('submission/v5_final/submission.geojson')
    
    # Remove Africa from v5
    non_africa = v5[v5.tile_id != TILE].copy()
    
    # Combine
    if len(africa_gdf) > 0:
        combined = gpd.pd.concat([non_africa, africa_gdf], ignore_index=True)
    else:
        combined = non_africa
    
    if 'id' in combined.columns:
        combined = combined.drop(columns=['id'])
    combined.insert(0, 'id', range(len(combined)))
    
    # Save
    out_dir = SUBMISSION / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'submission.geojson'
    combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
        out_path, driver='GeoJSON'
    )
    
    # Count per tile
    tc = combined.tile_id.value_counts()
    print(f'  {variant_name}: {len(combined)} total polys')
    for t in sorted(tc.index):
        print(f'    {t}: {tc[t]} polys')
    
    return out_path


def main():
    print("Loading data...")
    data = load_data()
    v5_prob = data['v5_prob']
    ekl_prob = data['ekl_prob']
    recent_loss = data['recent_loss']
    mk2_mask = data['mk2_mask']
    mk1_mask = data['mk1_mask']
    treecover = data['treecover']
    prof = data['prof']
    h, w = v5_prob.shape
    
    # ═══════════════════════════════════════════
    # Define Africa strategies
    # ═══════════════════════════════════════════
    
    strategies = {}
    
    # Strategy 1: CONSERVATIVE — raise threshold to 0.70
    # Rationale: Africa has no training data, so be more cautious
    # Removes marginal predictions, keeps only high-confidence ones
    mask = pp((v5_prob >= 0.70).astype(np.uint8))
    strategies['v6_africa_conservative'] = (mask, v5_prob)
    
    # Strategy 2: HANSEN-BOOSTED — v5(high) ∪ (v5(low) ∩ Hansen)
    # Rationale: Keep high-confidence v5 + add Hansen-confirmed areas at lower threshold
    # This adds predictions where an independent data source agrees
    v5_high = (v5_prob >= 0.70).astype(np.uint8)
    v5_low_hans = ((v5_prob >= 0.30) & recent_loss).astype(np.uint8)
    mask = pp((v5_high | v5_low_hans).astype(np.uint8))
    strategies['v6_africa_hansen_boost'] = (mask, v5_prob)
    
    # Strategy 3: CONSENSUS — v5(0.45) where ANY other source agrees
    # Rationale: Multi-source consensus is more robust for unseen regions
    any_agree = (recent_loss | mk2_mask | mk1_mask).astype(np.uint8)
    mask = pp(((v5_prob >= 0.45) & any_agree).astype(np.uint8))
    strategies['v6_africa_consensus'] = (mask, v5_prob)
    
    # Strategy 4: EMPTY — submit nothing for Africa
    # Rationale: if our Africa predictions are mostly FP, removing them helps
    # Zero FP but also zero TP for Africa
    mask = np.zeros((h, w), dtype=np.uint8)
    strategies['v6_africa_empty'] = (mask, v5_prob)
    
    # Strategy 5: HANSEN ONLY for Africa
    # Rationale: Hansen is actual forest loss observation, not a model extrapolation
    # Filter to high-treecover areas and apply morphology
    mask = pp((recent_loss & (treecover >= 50)).astype(np.uint8))
    strategies['v6_africa_hansen_only'] = (mask, v5_prob * 0.5 + 0.25)  # moderate confidence
    
    print(f"\n{'='*60}")
    print("Generating Africa variant submissions")
    print(f"{'='*60}\n")
    
    for name, (mask, prob) in strategies.items():
        print(f"\n--- {name} ---")
        print(f"  Africa mask: {mask.sum()} pixels")
        gdf = mask_to_geojson(mask, prob, prof, min_ha=0.5)
        print(f"  Africa polygons: {len(gdf)}")
        if len(gdf) > 0:
            utm_crs = gdf.estimate_utm_crs()
            area = gdf.to_crs(utm_crs).area.sum() / 10000
            print(f"  Africa area: {area:.0f} ha")
        build_submission(gdf, name)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':35s} {'Africa polys':>12s} {'Total polys':>11s}")
    print("-" * 60)
    for name in strategies:
        g = gpd.read_file(f'submission/{name}/submission.geojson')
        af = g[g.tile_id == TILE]
        print(f"{name:35s} {len(af):>12d} {len(g):>11d}")


if __name__ == "__main__":
    main()
