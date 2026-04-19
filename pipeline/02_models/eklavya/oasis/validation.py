"""Region-stratified leave-one-region-out (LORO) validation.

Tile MGRS prefixes:

* Amazon  : 18N*, 19N*  (Brazil/Peru/Colombia)
* SE Asia : 47Q*, 48P*, 48Q*  (Vietnam/Indonesia)
* Africa  : 33N*  (test only - no training analog!)

LORO splits hold out one whole region as validation. This is the only
honest proxy we have for the unseen Africa test biome.

We also expose a per-tile leave-one-tile-out generator for fine-grained
out-of-fold predictions used by the stacking ensemble.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

REGION_PREFIXES = {
    "amazon": ("18N", "19N"),
    "asia": ("47Q", "48P", "48Q"),
    "africa": ("33N",),
}


def region_of(tile_id: str) -> str:
    """Map a tile id like ``18NXH_6_8`` to its region label."""
    head = tile_id[:3]
    for region, prefixes in REGION_PREFIXES.items():
        if head in prefixes:
            return region
    return "unknown"


@dataclass
class Split:
    name: str
    train: list[str]
    val: list[str]


def loro_splits(train_tiles: list[str]) -> list[Split]:
    """One Split per training-region group held out at a time.

    Africa is in REGION_PREFIXES but never has training tiles, so it is
    naturally absent from the splits.
    """
    by_region: dict[str, list[str]] = {}
    for tile in train_tiles:
        by_region.setdefault(region_of(tile), []).append(tile)

    splits: list[Split] = []
    for region, val_tiles in sorted(by_region.items()):
        if region == "unknown":
            continue
        train_tiles_split = [t for t in train_tiles if region_of(t) != region]
        if not train_tiles_split:
            continue
        splits.append(Split(name=f"loro_{region}", train=sorted(train_tiles_split), val=sorted(val_tiles)))
    return splits


def loto_splits(train_tiles: list[str]) -> Iterator[Split]:
    """Leave-one-tile-out generator used for OOF prediction stacks."""
    for tile in train_tiles:
        yield Split(name=f"loto_{tile}", train=sorted(t for t in train_tiles if t != tile), val=[tile])


def region_summary(train_tiles: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tile in train_tiles:
        out.setdefault(region_of(tile), []).append(tile)
    return {k: sorted(v) for k, v in out.items()}
