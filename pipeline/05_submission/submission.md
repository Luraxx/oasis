# Submission Generation — Vectorization + Year Estimation

## Overview

The binary deforestation mask is converted to GeoJSON polygons with `time_step` (year of deforestation) and `confidence` attributes.

```
Binary mask (post-processed)
       ↓
① Raster → Polygon vectorization (rasterio.features.shapes)
       ↓
② Reproject UTM → WGS84 (EPSG:4326)
       ↓
③ Area filtering (≥ 0.10 ha in UTM meters)
       ↓
④ Year estimation (NBR trajectory analysis per polygon)
       ↓
⑤ Confidence assignment (fused probability at centroid)
       ↓
GeoJSON FeatureCollection → submission.geojson
```

## Vectorization

```python
from rasterio.features import shapes
from shapely.geometry import shape

# Extract polygon boundaries from binary raster
polygons = []
for geom, val in shapes(binary.astype(np.uint8), 
                        mask=binary.astype(bool), 
                        transform=profile['transform']):
    if val == 1:
        polygons.append(shape(geom))

# Create GeoDataFrame, reproject, filter by area
gdf = gpd.GeoDataFrame(geometry=polygons, crs=profile['crs'])
gdf = gdf.to_crs("EPSG:4326")
utm_crs = gdf.estimate_utm_crs()
areas_ha = gdf.to_crs(utm_crs).area / 10000.0
gdf = gdf[areas_ha >= 0.10].reset_index(drop=True)
```

## Year Estimation (NBR Trajectory)

For each deforestation polygon, estimate **when** the deforestation occurred:

1. Load S2 monthly data (B08/NIR + B12/SWIR) for 2020–2026
2. Compute NBR = (NIR − SWIR) / (NIR + SWIR) per month
3. Aggregate to yearly median NBR per pixel
4. Find year with **maximum NBR drop** (largest decrease = deforestation event)
5. Take drop-weighted modal year across all pixels in the polygon
6. Format as `YYMM` (e.g., "2406" = June 2024)

```python
def precompute_year_data(tile, profile):
    """Compute per-pixel year of max NBR drop."""
    yearly_nbr = {}
    for year in range(2020, 2026):
        monthly_nbr = []
        for month in range(1, 13):
            path = data_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if path.exists():
                b08, b12 = load_bands(path, [7, 11])
                nbr = (b08 - b12) / (b08 + b12)
                monthly_nbr.append(nbr)
        if monthly_nbr:
            yearly_nbr[year] = np.nanmedian(monthly_nbr, axis=0)
    
    # Find year of maximum drop for each pixel
    max_drop = np.zeros(shape)
    drop_year = np.full(shape, 2023)
    for i in range(1, len(sorted_years)):
        prev, curr = sorted_years[i-1], sorted_years[i]
        drop = yearly_nbr[prev] - yearly_nbr[curr]
        improved = drop > max_drop
        max_drop[improved] = drop[improved]
        drop_year[improved] = curr
    
    return drop_year, max_drop
```

## Output Format

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "time_step": "2406",
        "confidence": 0.654,
        "tile_id": "47QMA_6_2"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon, lat], [lon, lat], ...]]
      }
    }
  ]
}
```

## Final Statistics (Try 15 — Best Submission)

| Tile | Polygons | Area (ha) | Region |
|------|----------|-----------|--------|
| 18NVJ_1_6 | ~300 | ~1200 | Amazon |
| 18NYH_2_1 | ~200 | ~800 | Amazon |
| 33NTE_5_1 | ~150 | ~600 | Africa |
| 47QMA_6_2 | ~250 | ~1000 | Asia |
| 48PWA_0_6 | ~200 | ~850 | Asia |
| **Total** | **~1100** | **~4450** | |

## Source Code

| File | Purpose |
|------|---------|
| `oasis-luis-v5/scripts/13_v11_final_push.py` | vectorize() + precompute_year_data() |
| `oasis-eklavya-v2/oasis/predict.py` | Original vectorization logic |
| `oasis-luis-v5/scripts/14_v12_polygon_quality.py` | Erosion-enhanced vectorization |
