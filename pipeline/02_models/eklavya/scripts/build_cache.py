"""Build per-tile cache bundles for every train+test tile.

This is a thin wrapper over ``oasis.cache build`` so it shows up as a
script alongside the training entry points.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (sys.path bootstrap)

from oasis import audit, cache, paths
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tiles", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    paths.ensure_dirs()
    train_tiles, test_tiles = audit.audit(strict=False)

    todo: list[tuple[str, str]] = []
    if args.split in ("train", "both"):
        todo.extend((t, "train") for t in train_tiles)
    if args.split in ("test", "both"):
        todo.extend((t, "test") for t in test_tiles)
    if args.tiles:
        wanted = set(args.tiles.split(","))
        todo = [(t, s) for t, s in todo if t in wanted]

    for tile, split in tqdm(todo, desc="cache build"):
        try:
            cache.build_tile_cache(tile, split, force=args.force, verbose=not args.quiet)
        except Exception as exc:
            print(f"[cache][error] {tile} ({split}): {exc}")


if __name__ == "__main__":
    main()
