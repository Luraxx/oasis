"""Backwards-compatible thin shim for the old monolithic CLI.

The production pipeline lives in the ``oasis`` package:

* ``python -m oasis.audit``                  -> dataset preflight
* ``python scripts/build_cache.py``           -> per-tile reprojected cache
* ``python -m oasis.externals fetch``        -> WorldCover / Hansen / JRC
* ``python scripts/train_lgbm.py``            -> LightGBM (LORO + OOF)
* ``python scripts/train_tcn.py``             -> Temporal CNN (LORO + OOF)
* ``python scripts/train_unet.py``            -> Multi-temporal U-Net
* ``python scripts/fit_ensemble.py``          -> isotonic + stacker
* ``python scripts/infer_test.py``            -> submission GeoJSON
* ``python -m oasis.submission_check ...``    -> validate

This file remains so that older notebooks / cell magics that still call
``python train_and_predict.py`` keep working: it simply runs the
end-to-end LightGBM-only fast path (audit -> cache -> LightGBM ->
infer) without GPU dependencies. For the full multi-model ensemble use
the scripts above.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast LightGBM-only end-to-end path. For the full ensemble see scripts/."
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--samples-per-tile", type=int, default=20_000)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    args = parser.parse_args()

    py = sys.executable

    if args.audit_only:
        _run([py, "-m", "oasis.audit"])
        return

    if not args.skip_cache:
        _run([py, "scripts/build_cache.py", "--split", "both"])

    if not args.skip_train:
        _run(
            [
                py, "scripts/train_lgbm.py",
                "--samples-per-tile", str(args.samples_per_tile),
                "--n-estimators", str(args.n_estimators),
            ]
        )

    if not args.skip_predict:
        cmd = [
            py, "scripts/infer_test.py",
            "--min-area-ha", str(args.min_area_ha),
        ]
        if args.threshold is not None:
            cmd += ["--threshold", str(args.threshold)]
        _run(cmd)


if __name__ == "__main__":
    main()
