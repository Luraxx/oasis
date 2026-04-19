"""Per-tile hybrid: choose best source per tile, merge into one submission.

Strategy: keep v13_erode4 (proven 53.46% baseline) for tiles where it covers the
forest loss well, swap in a more aggressive cmr-aware source for under-predicted
tiles. Time_step is normalized to YYMM string (default '2306') and never null.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/shared-docker/oasis-mark-2")
V13 = Path("/shared-docker/oasis-luis-v5/submission/v13_shrink/erode4_raw/submission.geojson")

SOURCES = {
    "v13": V13,
    "tri_mega_max":   ROOT / "submissions/tri_mega_max/submission.geojson",
    "tri_mega_e4_p10": ROOT / "submissions/tri_mega_e4_p10/submission.geojson",
    "tri_mega_e6_p10": ROOT / "submissions/tri_mega_e6_p10/submission.geojson",
    "tri_mega_e6_p20": ROOT / "submissions/tri_mega_e6_p20/submission.geojson",
    "tri_mega_e8_p10": ROOT / "submissions/tri_mega_e8_p10/submission.geojson",
    "tri_mega_e8_p20": ROOT / "submissions/tri_mega_e8_p20/submission.geojson",
}


def fix_time_step(ts):
    if ts is None:
        return "2306"
    if isinstance(ts, str):
        return ts if len(ts) == 4 else "2306"
    s = str(int(ts))
    return s[2:] if len(s) == 6 else (s if len(s) == 4 else "2306")


def load_features_for_tile(src_name: str, tile: str):
    path = SOURCES[src_name]
    gj = json.load(open(path))
    out = []
    for f in gj["features"]:
        if f["properties"].get("tile_id") != tile:
            continue
        f = json.loads(json.dumps(f))  # deep copy
        f["properties"]["time_step"] = fix_time_step(f["properties"].get("time_step"))
        f["properties"]["tile_id"] = tile
        if "confidence" not in f["properties"]:
            f["properties"]["confidence"] = 0.5
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tile-18NVJ", default="v13")
    ap.add_argument("--tile-18NYH", default="v13")
    ap.add_argument("--tile-33NTE", default="v13")
    ap.add_argument("--tile-47QMA", default="v13")
    ap.add_argument("--tile-48PWA", default="v13")
    args = ap.parse_args()

    plan = {
        "18NVJ_1_6": args.tile_18NVJ,
        "18NYH_2_1": args.tile_18NYH,
        "33NTE_5_1": args.tile_33NTE,
        "47QMA_6_2": args.tile_47QMA,
        "48PWA_0_6": args.tile_48PWA,
    }
    print(f"=== {args.name} ===")
    for tid, src in plan.items():
        print(f"  {tid}: {src}")

    out_dir = ROOT / "submissions" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_feats = []
    per_tile = {}
    for tid, src in plan.items():
        feats = load_features_for_tile(src, tid)
        per_tile[tid] = {"src": src, "polys": len(feats)}
        all_feats.extend(feats)

    for i, f in enumerate(all_feats):
        f["id"] = str(i)
        f["properties"]["id"] = i

    cgj = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
        "features": all_feats,
    }
    out_path = out_dir / "submission.geojson"
    out_path.write_text(json.dumps(cgj))

    raw = out_path.read_bytes()
    assert raw[:1] == b"{", f"corrupt prefix: {raw[:8]!r}"
    parsed = json.loads(raw)
    assert parsed["type"] == "FeatureCollection"
    assert all(f["properties"].get("time_step") for f in parsed["features"])

    summary = {"plan": plan, "per_tile": per_tile, "total_polys": len(all_feats)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}  total_polys={len(all_feats)}")
    print("Validated: parses OK, all time_steps non-null")


if __name__ == "__main__":
    main()
