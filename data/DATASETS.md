# Data acquisition and label contract

No datasets, patient images, pretrained weights or clinical guideline text are redistributed.

| Dataset | Intended role | Official source / license action |
|---|---|---|
| Nutrition5k | Measured dish mass and dish nutrients; RGB-D geometry | https://github.com/google-research-datasets/Nutrition5k — review repository and dataset terms before downloading Google Cloud Storage shards |
| Food-101 | Crop classification | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ — research-use terms; images have third-party provenance |
| UEC-FOOD256 | Detection and classification | https://foodcam.mobi/dataset256.html — review research-use restrictions |
| FoodSeg103 | Pixel segmentation validation | https://github.com/LARC-CMU-SMU/FoodSeg103 — verify release-specific dataset and image licenses |
| Regional / Indian | Regional recognition | Select an explicitly licensed dataset; record original source, authors, version and permitted use; do not assume Kaggle availability grants redistribution rights |

The downloader verifies a supplied checksum and supports HTTP Range resume. No fabricated checksums or unofficial mirrors are supplied. Example:

```bash
python scripts/download_dataset.py food101 --accept-license --sha256 "$FOOD101_SHA256" --out data/raw/food101.tar.gz
```

Use the official Nutrition5k instructions to select RGB-D shards appropriate to Colab disk quotas; do not download every video. Safely unpack trusted archives into `data/raw`, inspect annotations and prepare JSONL records with these fields:

- `sample_id`: globally unique immutable sample identifier.
- `dataset`, `image`: source identifier and local image path.
- `dish_id`: shared across all frames, views and crops of the same physical dish; prefix IDs with dataset name.
- `label`: original dataset label.
- `bbox`: optional human annotated `[x1,y1,x2,y2]` in pixels.
- `weight_g`: measured ground truth for the SAME image scope, never guessed.
- `geometry_npz`: optional artifact with `depth_cm`, `mask`, `plate_mask`, scalar `cm_per_pixel`, scalar boolean `calibrated`.
- `food_id`: verified local nutrition match when available.

Taxonomy JSON maps `dataset:label` to `{ "name": "canonical food name", "fdc_id": null }`. A null ID means unresolved. Never silently map recipes to a single generic food. A domain reviewer must resolve label conflicts and match cooking method and edible portion to an actual FDC record.

`python -m ml.data` hashes source images, merges duplicate-image groups across dish IDs, rejects contradictory duplicate annotations, and assigns the same fixed-seed dish split (70/10/10/10 train/validation/calibration/test) before crop extraction. Exact duplicates are removed; near-duplicates still need perceptual or manual review. RF-DETR COCO exports use this same manifest. Classification uses inverse-frequency sampling only in training. PCA is fitted on training embeddings only. Calibration is separate from model selection and testing.

Nutrition5k provides ingredient weights and dish-level imagery, not ground-truth per-item masks. Train the CNN weight baseline at dish level using total dish mass. Evaluate dish-summed physics estimates only where aligned depth, calibrated geometry and plate/background masks are available. Do not divide dish weights among pseudo-masks and call them ground truth. Item-level evaluation requires a separate manually masked and individually weighed dataset. Dish-level class can be a reviewed dish label; per-item metrics cannot be claimed from dish-level labels.

The included CSV is deliberately labeled illustrative, NOT a validated USDA cache. FDC search is live when `FDC_API_KEY` is configured; otherwise it falls back to the local CSV. Missing purines, oxalate and some GI data remain unknown. An FDC ID, original response, retrieval timestamp, unit normalization and reviewer signoff are needed before replacing seed values. Source licensing for GI/purine/oxalate data must be checked separately.
