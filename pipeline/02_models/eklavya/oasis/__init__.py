"""Oasis deforestation-detection pipeline.

Package layout:

* paths       - canonical filesystem paths for data, cache, externals, outputs.
* audit       - dataset preflight against metadata.
* cache       - per-tile reprojected feature/label cache on scratch SSD.
* labels      - weak-label loaders + consensus fusion policy.
* features    - per-modality feature extractors (S1, S2, AEF, externals).
* validation  - leave-one-region-out tile splits, OOF accumulator.
* models      - LightGBM / Temporal CNN / U-Net trainers.
* ensemble    - calibration + stacking.
* postprocess - morphology + connected-component filtering.
* predict     - test-tile inference utilities.
"""

from oasis import paths  # noqa: F401
