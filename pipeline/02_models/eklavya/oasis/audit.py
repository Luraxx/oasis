"""Dataset preflight audit.

Compares metadata tile lists against on-disk data for S1, S2, AEF, and
all weak-label sources. Reports missing tiles, missing modalities, and
suspicious file counts. Exits non-zero in ``--strict`` mode if anything
is missing.

Usage:
    python -m oasis.audit
    python -m oasis.audit --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from oasis import paths


def load_metadata_tiles(split: str) -> list[str]:
    meta_path = paths.META_DIR / f"{split}_tiles.geojson"
    with open(meta_path) as f:
        meta = json.load(f)
    return sorted(feature["properties"]["name"] for feature in meta["features"])


def discover_split_tiles(base_dir: Path, split: str, suffix: str) -> set[str]:
    split_dir = base_dir / split
    if not split_dir.exists():
        return set()
    return {p.name.replace(suffix, "") for p in split_dir.iterdir() if p.is_dir()}


def discover_aef_tiles(split: str) -> set[str]:
    split_dir = paths.AEF_DIR / split
    if not split_dir.exists():
        return set()
    return {p.stem.rsplit("_", 1)[0] for p in split_dir.glob("*.tiff")}


def discover_label_tiles(label_name: str) -> set[str]:
    label_dir = paths.LABEL_DIR / label_name
    if not label_dir.exists():
        return set()
    out: set[str] = set()
    for p in label_dir.glob("*.tif"):
        stem = p.stem
        if label_name == "radd":
            out.add(stem.replace("radd_", "").replace("_labels", ""))
        elif label_name == "glads2":
            out.add(stem.replace("glads2_", "").split("_alert")[0])
        elif label_name == "gladl":
            out.add(stem.replace("gladl_", "").split("_alert")[0])
    return out


def per_tile_file_counts(split_dir: Path) -> dict[str, int]:
    if not split_dir.exists():
        return {}
    return {p.name: len(list(p.glob("*.tif"))) for p in split_dir.iterdir() if p.is_dir()}


def print_count_summary(name: str, counts: dict[str, int]) -> None:
    if not counts:
        print(f"{name}: missing")
        return
    values = np.array(list(counts.values()))
    print(
        f"{name}: tiles={len(values)}, min={values.min()}, "
        f"median={np.median(values):.0f}, max={values.max()}"
    )


def audit(strict: bool = False) -> tuple[list[str], list[str]]:
    """Run the audit. Returns (usable_train_tiles, usable_test_tiles).

    Raises SystemExit(1) under strict mode if mandatory tiles or
    modalities are missing.
    """
    expected_train = load_metadata_tiles("train")
    expected_test = load_metadata_tiles("test")

    s2_train = discover_split_tiles(paths.S2_DIR, "train", "__s2_l2a")
    s2_test = discover_split_tiles(paths.S2_DIR, "test", "__s2_l2a")
    s1_train = discover_split_tiles(paths.S1_DIR, "train", "__s1_rtc")
    s1_test = discover_split_tiles(paths.S1_DIR, "test", "__s1_rtc")
    aef_train = discover_aef_tiles("train")
    aef_test = discover_aef_tiles("test")

    label_presence = {name: discover_label_tiles(name) for name in paths.LABEL_SOURCES}

    print("=" * 60)
    print("DATASET PREFLIGHT")
    print("=" * 60)
    print(f"Metadata train tiles: {len(expected_train)}")
    print(f"Metadata test tiles : {len(expected_test)}")
    print(f"S2 train/test       : {len(s2_train)} / {len(s2_test)}")
    print(f"S1 train/test       : {len(s1_train)} / {len(s1_test)}")
    print(f"AEF train/test      : {len(aef_train)} / {len(aef_test)}")
    print(
        "Label train coverage: "
        + ", ".join(f"{k}={len(v)}" for k, v in label_presence.items())
    )

    issues: list[str] = []

    missing_train_s2 = sorted(set(expected_train) - s2_train)
    missing_test_s2 = sorted(set(expected_test) - s2_test)
    if missing_train_s2:
        issues.append(f"missing S2 train: {missing_train_s2}")
    if missing_test_s2:
        issues.append(f"missing S2 test: {missing_test_s2}")

    print_count_summary("S2 train file counts", per_tile_file_counts(paths.S2_DIR / "train"))
    print_count_summary("S1 train file counts", per_tile_file_counts(paths.S1_DIR / "train"))
    print_count_summary("S2 test file counts ", per_tile_file_counts(paths.S2_DIR / "test"))
    print_count_summary("S1 test file counts ", per_tile_file_counts(paths.S1_DIR / "test"))

    train_tiles = sorted(set(expected_train) & s2_train)
    test_tiles = sorted(set(expected_test) & s2_test)

    for tile in train_tiles:
        if tile not in s1_train:
            issues.append(f"train tile {tile} missing S1")
        if tile not in aef_train:
            issues.append(f"train tile {tile} missing AEF")
        if tile not in label_presence["radd"] and tile not in label_presence["gladl"]:
            issues.append(f"train tile {tile} has no RADD nor GLAD-L coverage")

    for tile in test_tiles:
        if tile not in s1_test:
            issues.append(f"test tile {tile} missing S1")
        if tile not in aef_test:
            issues.append(f"test tile {tile} missing AEF")

    if issues:
        print("\nISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues found.")

    print(f"\nUsable train tiles: {train_tiles}")
    print(f"Usable test tiles : {test_tiles}")

    if strict and issues:
        raise SystemExit(1)

    return train_tiles, test_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset coverage.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on any issue.")
    args = parser.parse_args()
    audit(strict=args.strict)


if __name__ == "__main__":
    main()
