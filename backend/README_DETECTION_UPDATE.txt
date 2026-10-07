GoatVision Detection Recall V1

Only the detection stage was changed. Existing annotation, mask rendering, ID rendering, weight, BCS, side-view, and SAM2 refinement logic are preserved.

New detection flow:
1. Project detector 640px
2. SAME project detector 1536px on every image
3. Project detector 1024px overlapping tiles for large images
4. Generic segmentation detector as additional recovery source
5. Merge + deduplicate
6. Existing SAM2 refinement
7. Existing confidence/mask safety filters

Important: finding one goat no longer stops detection.

Current final safety thresholds remain unchanged: MIN_VALID_DET_CONF and MIN_VALID_MASK_FRACTION.

The file is intended to replace the current backend engine file only after making a backup.
