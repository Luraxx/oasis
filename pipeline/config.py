"""
Pipeline-Konfiguration: Pfade, Tiles, und Hyperparameter.

Alle Pfade sind relativ zum Repository-Root aufgebaut.
Passe DATA_ROOT an wenn die Daten woanders liegen.
"""
from pathlib import Path

# ── Wurzelverzeichnis ───────────────────────────────────────────────
# REPO_ROOT = Ordner in dem dieses Repo liegt (eins über pipeline/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

# ── Probability Maps (Modell-Outputs) ──────────────────────────────
# Eklavya: 5-Modell-Ensemble (3×LightGBM + TCN + UNet), Leave-One-Region-Out
# Dateien: prob_18NVJ_1_6.tif, prob_18NYH_2_1.tif, ... (5 Tiles)
EKL_PROBS = DATA_ROOT / "ekl_submission"

# Luis: LightGBM + UNet Ensemble, Leave-One-Region-Out
LUIS_PROBS = DATA_ROOT / "luis_v4_submission"

# ── Sentinel-2 Daten (für Year-Estimation) ─────────────────────────
# Challenge-Daten: monatliche S2-Mediane 2020-2025
S2_DATA = DATA_ROOT / "makeathon-challenge" / "sentinel-2" / "test"
# Zusätzliche Daten: S2-Mediane Jan-Apr 2026
S2_ADDITIONAL = DATA_ROOT / "sentinel-additional" / "sentinel-2"

# ── Output ──────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "pipeline" / "output"

# ── Test-Tiles ──────────────────────────────────────────────────────
TEST_TILES = [
    "18NVJ_1_6",   # Amazonas, Brasilien     (EPSG:32618)
    "18NYH_2_1",   # Amazonas, Brasilien     (EPSG:32618)
    "33NTE_5_1",   # Kamerun, Afrika         (EPSG:32633)
    "47QMA_6_2",   # Borneo, Südostasien     (EPSG:32647)
    "48PWA_0_6",   # Sumatra, Südostasien    (EPSG:32648)
]

# ── Ensemble-Gewichte ──────────────────────────────────────────────
# 60% Eklavya + 40% Luis ergibt die beste Fusion.
# Begründung: Eklavya hat höhere Einzelmodell-Qualität (5er-Stack),
# Luis bringt komplementäre Informationen (andere Feature-Auswahl).
EKL_WEIGHT = 0.60
LUIS_WEIGHT = 0.40

# ── Schwellenwerte pro Tile ────────────────────────────────────────
# "Ultra-low" Thresholds: aggressiv niedrig um maximale Recall zu erreichen.
# Die hohe FPR wird anschließend durch Morphologie (Erosion) korrigiert.
THRESHOLDS = {
    "18NVJ_1_6": 0.20,   # Amazonas → viele kleine Patches → niedrig
    "18NYH_2_1": 0.28,   # Amazonas → größere zusammenhängende Flächen
    "33NTE_5_1": 0.25,   # Afrika → moderates Vertrauen
    "47QMA_6_2": 0.20,   # Borneo → schwieriges Terrain → niedrig
    "48PWA_0_6": 0.28,   # Sumatra → ähnlich wie NYH
}

# ── Morphologie-Parameter ──────────────────────────────────────────
# Schritt 1: Close(2) — verbindet nahe Fragmente (8-connected, 2 Iterationen)
CLOSE_ITERATIONS = 2

# Schritt 2: Dilate(2) — expandiert Polygon-Ränder um ~20m
#   → fängt Grenzpixel ein die von der Binarisierung abgeschnitten wurden
DILATE_ITERATIONS = 2

# Schritt 3: Erode(4) — erodiert um ~40m zurück
#   → entfernt False Positives an Rändern, verbessert Polygon-Qualität
#   → Netto-Effekt: Boundaries werden geglättet, kleine FP verschwinden
#   → Das war der Schlüssel-Move von Try 11 (52.64%) → Try 15 (53.46%)
ERODE_ITERATIONS = 4

# ── Flächenfilter ──────────────────────────────────────────────────
# Polygone kleiner als 0.10 ha (= 10 Pixel à 10×10m) werden entfernt.
MIN_AREA_HA = 0.10
