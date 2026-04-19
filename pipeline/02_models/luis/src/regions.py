"""Region detection from MGRS tile names."""

REGION_PREFIXES = {
    "amazon": ("18N", "19N"),
    "asia":   ("47Q", "48P", "48Q"),
    "africa": ("33N",),  # test only — no training tiles
}


def tile_region(tile: str) -> str:
    """Return 'amazon' / 'asia' / 'africa' for a given MGRS tile name."""
    for region, prefixes in REGION_PREFIXES.items():
        for p in prefixes:
            if tile.startswith(p):
                return region
    return "unknown"
