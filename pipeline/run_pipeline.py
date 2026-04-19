#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  OASIS Deforestation Detection Pipeline — Reproduziert Try 15 (53.46% IoU)
  Team: OASIS | osapiens Makeathon 2026 | 4. Platz
═══════════════════════════════════════════════════════════════════════════

Dieses Script reproduziert unsere beste Submission (Try 15, 53.46% Union IoU)
in einem einzigen Durchlauf. Es nimmt die vortrainierten Probability Maps
der Einzelmodelle und erzeugt daraus eine submission.geojson.

VORAUSSETZUNG: Die Probability Maps müssen existieren (siehe config.py).
               Diese werden von den Einzelmodellen in pipeline/02_models/ erzeugt.

PIPELINE-SCHRITTE:
  1. LADEN         — Probability Maps von Eklavya + Luis laden
  2. FUSION        — Gewichteter Durchschnitt (60/40)
  3. BINARISIERUNG — Per-Tile Schwellenwerte anwenden
  4. MORPHOLOGIE   — Close → Dilate → Erode (Polygon-Qualität)
  5. YEAR-ESTIMATION — NBR-Drop-Analyse auf Sentinel-2 Zeitreihe
  6. VEKTORISIERUNG — Raster → Polygone als GeoJSON

Aufruf:
  python run_pipeline.py                  # Standard: Try 15 Konfiguration
  python run_pipeline.py --no-erosion     # Ohne Erosion (Try 11 Basis)
  python run_pipeline.py --dry-run        # Nur Statistiken, kein Output
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from scipy import ndimage
from rasterio.features import shapes
from shapely.geometry import shape

import config as cfg

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════
# SCHRITT 1: PROBABILITY MAPS LADEN
# ═════════════════════════════════════════════════════════════════════════
# Jedes Modell-Team hat pro Tile eine Probability Map erzeugt:
#   prob_TILE.tif — Float32, Werte 0-1000 (werden auf 0.0-1.0 normiert)
#   Jeder Pixel = P(Entwaldung) an dieser 10×10m Stelle
#
# Die Maps wurden unabhängig trainiert:
#   - Eklavya: 508 Features → 5-Modell-Stack (3×LGBM + TCN + UNet)
#   - Luis:    508 Features → LGBM + UNet, Leave-One-Region-Out
# ═════════════════════════════════════════════════════════════════════════

def load_prob(path: Path) -> tuple[np.ndarray, dict]:
    """Lädt eine Probability Map und normiert auf [0, 1]."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    # Probability Maps sind als 0-1000 Integer gespeichert → auf 0.0-1.0
    if data.max() > 10:
        data /= 1000.0
    return data, profile


def load_tile(tile: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Lädt Eklavya + Luis Probability Maps für ein Tile."""
    ekl, profile = load_prob(cfg.EKL_PROBS / f"prob_{tile}.tif")

    luis_path = cfg.LUIS_PROBS / f"prob_{tile}.tif"
    if luis_path.exists():
        luis, _ = load_prob(luis_path)
        # Sicherstellen dass beide die gleiche Größe haben
        h = min(ekl.shape[0], luis.shape[0])
        w = min(ekl.shape[1], luis.shape[1])
        ekl, luis = ekl[:h, :w], luis[:h, :w]
    else:
        luis = None

    return ekl, luis, profile


# ═════════════════════════════════════════════════════════════════════════
# SCHRITT 2: FUSION — Gewichteter Durchschnitt
# ═════════════════════════════════════════════════════════════════════════
# Wir kombinieren die Vorhersagen beider Modelle durch gewichteten
# Durchschnitt: P_fused = 0.6 × P_ekl + 0.4 × P_luis
#
# Warum 60/40?
#   - Eklavya hat das stärkere Einzelmodell (5er-Ensemble)
#   - Aber Luis bringt komplementäre Info (andere Regularisierung)
#   - 60/40 war konsistent besser als 50/50 oder 70/30 in OOF-Validierung
# ═════════════════════════════════════════════════════════════════════════

def fuse(ekl: np.ndarray, luis: np.ndarray | None) -> np.ndarray:
    """Fusioniert zwei Probability Maps durch gewichteten Durchschnitt."""
    if luis is None:
        return ekl
    return cfg.EKL_WEIGHT * ekl + cfg.LUIS_WEIGHT * luis


# ═════════════════════════════════════════════════════════════════════════
# SCHRITT 3 + 4: BINARISIERUNG + MORPHOLOGIE
# ═════════════════════════════════════════════════════════════════════════
# 3) Binarisierung: Pixel mit P ≥ threshold werden als "entwaldet" markiert.
#    Die Thresholds sind absichtlich NIEDRIG (0.20-0.28) um hohe Recall
#    zu erreichen. Das erzeugt viele False Positives — die werden in
#    Schritt 4 durch Morphologie wieder entfernt.
#
# 4) Morphologie-Kette:
#    a) Close(2)  — verbindet nahe Fragmente (Kernel: 8-connected, 2 Iter.)
#                   → Kleine Lücken in Entwaldungsflächen werden geschlossen
#    b) Dilate(2) — expandiert alle Polygone um ~20m nach außen
#                   → Fängt Grenzpixel ein die knapp unter dem Threshold lagen
#    c) Erode(4)  — erodiert um ~40m zurück
#                   → SCHLÜSSEL-SCHRITT: Netto-Schrumpfung von ~20m
#                   → Entfernt dünne False-Positive-Streifen an Rändern
#                   → Kleine isolierte FP-Cluster verschwinden komplett
#                   → Boundaries werden glatter (bessere Polygon-Qualität)
#
#    Warum Dilate+Erode statt nur Erode?
#      → Dilate zuerst verbindet nahe Cluster zu einem Polygon
#      → Erode danach entfernt nur die äußeren Ränder
#      → Ergebnis: Kompakte, gut geformte Polygone
# ═════════════════════════════════════════════════════════════════════════

def binarize_and_morph(
    prob: np.ndarray,
    threshold: float,
    close_iter: int,
    dilate_iter: int,
    erode_iter: int,
    min_area_ha: float,
) -> np.ndarray:
    """Binarisiert die Probability Map und wendet Morphologie an."""
    # Schritt 3: Binarisierung
    binary = (prob >= threshold).astype(np.uint8)

    # Schritt 4a: Closing — 8-connected Kernel, verbindet Fragmente
    if close_iter > 0:
        kernel_8conn = ndimage.generate_binary_structure(2, 2)
        binary = ndimage.binary_closing(
            binary, kernel_8conn, iterations=close_iter
        ).astype(np.uint8)

    # Schritt 4b: Dilation — 4-connected Kernel, expandiert Ränder
    if dilate_iter > 0:
        kernel_4conn = ndimage.generate_binary_structure(2, 1)
        binary = ndimage.binary_dilation(
            binary, kernel_4conn, iterations=dilate_iter
        ).astype(np.uint8)

    # Schritt 4c: Erosion — 4-connected Kernel, schrumpft zurück
    if erode_iter > 0:
        kernel_4conn = ndimage.generate_binary_structure(2, 1)
        binary = ndimage.binary_erosion(
            binary, kernel_4conn, iterations=erode_iter
        ).astype(np.uint8)

    # Flächenfilter: Kleine Cluster entfernen
    labeled, n_components = ndimage.label(binary)
    min_pixels = int(min_area_ha / 0.01)  # 1 Pixel = 10m × 10m = 0.01 ha
    for component_id in range(1, n_components + 1):
        component_mask = labeled == component_id
        if component_mask.sum() < min_pixels:
            binary[component_mask] = 0

    return binary


# ═════════════════════════════════════════════════════════════════════════
# SCHRITT 5: YEAR ESTIMATION — Wann wurde entwaldet?
# ═════════════════════════════════════════════════════════════════════════
# Für jedes erkannte Entwaldungs-Polygon müssen wir schätzen WANN die
# Entwaldung stattfand. Das machen wir über NBR (Normalized Burn Ratio):
#
#   NBR = (B08 - B12) / (B08 + B12)
#
# NBR ist hoch für gesunde Vegetation und fällt stark ab bei Entwaldung.
# Wir berechnen den jährlichen Median-NBR für 2020-2026 und finden das
# Jahr mit dem größten NBR-Drop → das ist das geschätzte Entwaldungsjahr.
#
# Format: "YYMM" z.B. "2306" = Juni 2023 (Default wenn kein Drop erkannt)
# ═════════════════════════════════════════════════════════════════════════

def estimate_deforestation_years(
    tile: str, profile: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet pro Pixel das geschätzte Entwaldungsjahr und die Drop-Stärke.

    Returns:
        drop_year: Array mit dem Jahr des stärksten NBR-Drops pro Pixel
        max_drop:  Array mit der Stärke des NBR-Drops (0 = kein Drop)
    """
    h, w = profile["height"], profile["width"]

    # Sentinel-2 Daten laden
    s2_dir = cfg.S2_DATA / f"{tile}__s2_l2a"
    if not s2_dir.exists():
        return np.full((h, w), 2023, dtype=np.int32), np.zeros((h, w), np.float32)

    # Jährlichen Median-NBR berechnen
    yearly_nbr = {}
    for year in range(2020, 2026):
        monthly_nbr = []
        for month in range(1, 13):
            tif_path = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if not tif_path.exists():
                continue
            try:
                with rasterio.open(tif_path) as src:
                    bands = src.read().astype(np.float32)
                b08, b12 = bands[7], bands[11]  # NIR, SWIR2
                denom = b08 + b12
                nbr = np.where(denom > 0, (b08 - b12) / denom, 0)
                nbr[bands[0] <= 0] = np.nan  # Nodata-Pixel maskieren
                monthly_nbr.append(nbr)
            except Exception:
                continue
        if monthly_nbr:
            mh = min(m.shape[0] for m in monthly_nbr)
            mw = min(m.shape[1] for m in monthly_nbr)
            monthly_nbr = [m[:mh, :mw] for m in monthly_nbr]
            with np.errstate(all="ignore"):
                yearly_nbr[year] = np.nanmedian(monthly_nbr, axis=0)

    # 2026 Daten (zusätzlich heruntergeladen, Jan-Apr)
    s2_add = cfg.S2_ADDITIONAL / f"{tile}__s2_l2a"
    if s2_add.exists():
        monthly_2026 = []
        for month in [1, 2, 3, 4]:
            tif_path = s2_add / f"{tile}__s2_l2a_2026_{month}.tif"
            if not tif_path.exists():
                continue
            try:
                with rasterio.open(tif_path) as src:
                    bands = src.read().astype(np.float32)
                b08, b12 = bands[7], bands[11]
                denom = b08 + b12
                nbr = np.where(denom > 0, (b08 - b12) / denom, 0)
                nbr[bands[0] <= 0] = np.nan
                monthly_2026.append(nbr)
            except Exception:
                continue
        if monthly_2026:
            mh = min(m.shape[0] for m in monthly_2026)
            mw = min(m.shape[1] for m in monthly_2026)
            monthly_2026 = [m[:mh, :mw] for m in monthly_2026]
            with np.errstate(all="ignore"):
                yearly_nbr[2026] = np.nanmedian(monthly_2026, axis=0)

    if len(yearly_nbr) < 2:
        return np.full((h, w), 2023, dtype=np.int32), np.zeros((h, w), np.float32)

    # Alle Arrays auf gleiche Größe bringen
    min_h = min(min(a.shape[0] for a in yearly_nbr.values()), h)
    min_w = min(min(a.shape[1] for a in yearly_nbr.values()), w)
    yearly_nbr = {k: v[:min_h, :min_w] for k, v in yearly_nbr.items()}

    # Größten NBR-Drop zwischen aufeinanderfolgenden Jahren finden
    sorted_years = sorted(yearly_nbr.keys())
    max_drop = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)

    for i in range(1, len(sorted_years)):
        prev_year, curr_year = sorted_years[i - 1], sorted_years[i]
        with np.errstate(all="ignore"):
            drop = np.nan_to_num(
                yearly_nbr[prev_year] - yearly_nbr[curr_year], nan=0.0
            )
        improved = drop > max_drop[:min_h, :min_w]
        max_drop[:min_h, :min_w][improved] = drop[improved]
        drop_year[:min_h, :min_w][improved] = curr_year

    return drop_year, max_drop


# ═════════════════════════════════════════════════════════════════════════
# SCHRITT 6: VEKTORISIERUNG — Raster → GeoJSON Polygone
# ═════════════════════════════════════════════════════════════════════════
# Konvertiert die binäre Maske in GeoJSON-Polygone mit:
#   - geometry: Polygon in EPSG:4326 (WGS84, Lat/Lon)
#   - time_step: "YYMM" Format, geschätztes Entwaldungsjahr
#   - confidence: 1.0 (wird nicht evaluiert)
#   - tile_id: Name des Tiles
# ═════════════════════════════════════════════════════════════════════════

def vectorize(
    binary: np.ndarray,
    profile: dict,
    tile: str,
    min_area_ha: float,
    drop_year: np.ndarray,
    max_drop: np.ndarray,
) -> gpd.GeoDataFrame:
    """Konvertiert binäre Maske in GeoDataFrame mit Polygonen."""
    empty = gpd.GeoDataFrame(
        columns=["geometry", "time_step", "confidence", "tile_id"]
    )
    if binary.sum() == 0:
        return empty

    # Raster → Polygone (im CRS des Rasters, also UTM)
    polys = []
    for geom, val in shapes(
        binary.astype(np.uint8),
        mask=binary.astype(bool),
        transform=profile["transform"],
    ):
        if val == 1:
            polys.append(shape(geom))
    if not polys:
        return empty

    # GeoDataFrame in UTM erstellen, dann nach WGS84 reprojizieren
    gdf = gpd.GeoDataFrame(geometry=polys, crs=profile["crs"])
    gdf = gdf.to_crs("EPSG:4326")

    # Flächenfilter in metrischen Koordinaten
    utm_crs = gdf.estimate_utm_crs()
    areas_ha = gdf.to_crs(utm_crs).area / 10_000.0
    gdf = gdf[areas_ha >= min_area_ha].reset_index(drop=True)
    if gdf.empty:
        return empty

    # ── Year-Estimation pro Polygon ──
    # Für jedes Polygon: Finde das dominante Entwaldungsjahr
    # basierend auf dem NBR-Drop der enthaltenen Pixel
    h, w = binary.shape
    labeled, n_components = ndimage.label(binary)
    year_map = np.full((h, w), "2306", dtype="<U6")  # Default: Juni 2023

    for cid in range(1, n_components + 1):
        mask = labeled == cid
        pixel_years = drop_year[mask]
        pixel_drops = max_drop[mask]
        if len(pixel_years) == 0:
            continue
        # Gewichtetes Modal-Jahr: Jahr mit der größten Summe an NBR-Drops
        if pixel_drops.sum() > 0:
            year_totals = {}
            for y, d in zip(pixel_years, pixel_drops):
                year_totals[y] = year_totals.get(y, 0) + d
            dominant_year = max(year_totals, key=year_totals.get)
        else:
            vals, cnts = np.unique(pixel_years, return_counts=True)
            dominant_year = vals[np.argmax(cnts)]
        year_map[mask] = f"{dominant_year % 100:02d}06"

    # Year-Stamp auf Polygone übertragen (via Centroid-Lookup)
    gdf["time_step"] = "2306"
    gdf["confidence"] = 1.0
    gdf["tile_id"] = tile

    gdf_projected = gdf.to_crs(profile["crs"])
    transform = profile["transform"]
    for idx, row in gdf_projected.iterrows():
        centroid = row.geometry.centroid
        col = int((centroid.x - transform.c) / transform.a)
        rpx = int((centroid.y - transform.f) / transform.e)
        if 0 <= rpx < h and 0 <= col < w:
            ts = year_map[rpx, col]
            if ts:
                gdf.at[idx, "time_step"] = ts

    return gdf


# ═════════════════════════════════════════════════════════════════════════
# SCHRITT 7: SUBMISSION ERZEUGEN
# ═════════════════════════════════════════════════════════════════════════

def create_submission(all_gdfs: list[gpd.GeoDataFrame], output_path: Path):
    """Kombiniert alle Tile-GeoDataFrames und schreibt submission.geojson."""
    if not all_gdfs:
        print("  WARNUNG: Keine Polygone gefunden!")
        return

    combined = gpd.pd.concat(all_gdfs, ignore_index=True)
    combined.insert(0, "id", range(len(combined)))
    columns = ["id", "time_step", "confidence", "tile_id", "geometry"]
    combined[columns].to_file(output_path, driver="GeoJSON")
    print(f"\n  ✓ Submission geschrieben: {output_path}")
    print(f"    {len(combined)} Polygone total")


# ═════════════════════════════════════════════════════════════════════════
# MAIN — Pipeline zusammenbauen und ausführen
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OASIS Deforestation Pipeline — reproduziert Try 15 (53.46% IoU)"
    )
    parser.add_argument(
        "--no-erosion", action="store_true",
        help="Erosion überspringen (reproduziert Try 11 Basis, 52.64%% IoU)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Nur Statistiken anzeigen, kein GeoJSON schreiben"
    )
    parser.add_argument(
        "--tiles", nargs="+", default=cfg.TEST_TILES,
        help="Nur bestimmte Tiles verarbeiten"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output-Pfad für submission.geojson"
    )
    args = parser.parse_args()

    erode_iter = 0 if args.no_erosion else cfg.ERODE_ITERATIONS
    variant = "try11_basis" if args.no_erosion else "try15_best"

    if args.output:
        out_dir = args.output
    else:
        out_dir = cfg.OUTPUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  OASIS Pipeline — {variant.upper()}")
    print(f"  Fusion: {cfg.EKL_WEIGHT:.0%} Eklavya + {cfg.LUIS_WEIGHT:.0%} Luis")
    print(f"  Morphologie: close={cfg.CLOSE_ITERATIONS}, "
          f"dilate={cfg.DILATE_ITERATIONS}, erode={erode_iter}")
    print(f"  Tiles: {len(args.tiles)}")
    print("=" * 72)

    t0 = time.time()
    all_gdfs = []
    total_stats = {"polys": 0, "area_ha": 0.0}

    for tile in args.tiles:
        t1 = time.time()
        print(f"\n── Tile: {tile} ──")

        # Schritt 1: Laden
        print("  [1/6] Probability Maps laden...")
        ekl, luis, profile = load_tile(tile)
        print(f"        Eklavya: {ekl.shape}, Luis: {'N/A' if luis is None else luis.shape}")

        # Schritt 2: Fusion
        print(f"  [2/6] Fusion ({cfg.EKL_WEIGHT:.0%}/{cfg.LUIS_WEIGHT:.0%})...")
        prob = fuse(ekl, luis)
        print(f"        Fused range: [{prob.min():.3f}, {prob.max():.3f}]")

        # Schritt 3+4: Binarisierung + Morphologie
        threshold = cfg.THRESHOLDS[tile]
        print(f"  [3/6] Binarisierung (threshold={threshold:.2f})...")
        print(f"  [4/6] Morphologie (close={cfg.CLOSE_ITERATIONS}, "
              f"dilate={cfg.DILATE_ITERATIONS}, erode={erode_iter})...")
        binary = binarize_and_morph(
            prob, threshold,
            cfg.CLOSE_ITERATIONS, cfg.DILATE_ITERATIONS,
            erode_iter, cfg.MIN_AREA_HA,
        )
        n_pixels = binary.sum()
        print(f"        {n_pixels:,} Pixel als entwaldet markiert "
              f"({n_pixels * 0.01:.1f} ha)")

        # Schritt 5: Year Estimation
        print("  [5/6] Year Estimation (NBR-Drop-Analyse)...")
        drop_year, max_drop = estimate_deforestation_years(tile, profile)
        unique_years = np.unique(drop_year[binary.astype(bool)])
        print(f"        Erkannte Jahre: {sorted(unique_years)}")

        # Schritt 6: Vektorisierung
        print("  [6/6] Vektorisierung...")
        gdf = vectorize(binary, profile, tile, cfg.MIN_AREA_HA, drop_year, max_drop)

        # Statistiken
        if len(gdf) > 0:
            utm = gdf.estimate_utm_crs()
            area_ha = gdf.to_crs(utm).area.sum() / 10_000.0
        else:
            area_ha = 0.0

        elapsed = time.time() - t1
        print(f"        → {len(gdf)} Polygone, {area_ha:.1f} ha ({elapsed:.1f}s)")

        total_stats["polys"] += len(gdf)
        total_stats["area_ha"] += area_ha
        if len(gdf) > 0:
            all_gdfs.append(gdf)

    # Zusammenfassung
    print(f"\n{'=' * 72}")
    print(f"  ERGEBNIS: {total_stats['polys']} Polygone, "
          f"{total_stats['area_ha']:.1f} ha total")
    print(f"  Gesamtzeit: {time.time() - t0:.1f}s")
    print(f"{'=' * 72}")

    # Schritt 7: Submission schreiben
    if not args.dry_run:
        create_submission(all_gdfs, out_dir / "submission.geojson")
    else:
        print("\n  (Dry-run: Kein Output geschrieben)")


if __name__ == "__main__":
    main()
