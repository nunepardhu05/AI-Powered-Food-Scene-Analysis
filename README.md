# NutriScope

A **local-first research application**, built with Python **3.12.14**, FastAPI, SQLite and a buildless HTML/CSS/JavaScript interface. No Node build is needed to run the app. This repository is a working software prototype, **not a completed or clinically validated food-estimation system**.

## Run in one command

From this directory:

```bash
docker compose up --build
```

Open **http://localhost:8000**. Compose binds only to loopback and persists profiles/history in the `nutrition-data` Docker volume. The container runs as a non-root user with a read-only filesystem. The first build needs Internet access to install the fully pinned Python lockfile. No trained ML models are bundled. Manual logging, guidance, meal calculations and history work without them.

To run without Docker:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`requirements.txt` pins direct application/test dependencies; `requirements.lock` pins **all** resolved transitive dependencies used in the verified Python environment. `requirements-ml.txt` and `requirements-ml.lock` do the same for the optional training stack. The ML lock is dependency-resolved, not a completed GPU installation/training validation. Do not substitute package versions without rerunning checks. The browser QA helper uses Playwright 1.62.0 and the environment's Chromium; it is not needed at runtime.

## What is implemented

- Responsive dashboard, light/dark appearance, profile wizard, searchable 155-condition catalog, optional labs/medications, clinician target minimums/maximums, image upload/camera, canvas overlays, editable food portions, serving percentages, journal, weekly logged-energy chart, browser-tab reminders and persistent disclaimer.
- Local SQLite profile and meal CRUD, nutrient intervals, BMI, adult maintenance-energy estimate, household conversions, ingredient/allergen exclusions, clinician overrides, stricter maximum / higher minimum merging and infeasible-bound detection.
- Versioned YAML files per condition, structural validator, template, explicit draft status, guideline source links, mapped category-level ICD-10-CM codes where supplied, unmapped code notices, free-text retention and suggestions.
- Tier 2 supervision banners and planning gates; Tier 3 food logging and avoid/allergen warnings, with no automatic energy target, weight-loss plan or diet restriction output. Pediatric, pregnancy and high-risk free text get additional gates.
- A constrained **illustrative** five-slot menu with distinct meals, preference-filtered options, and up to three individually checked swaps per slot. It excludes known allergens and enforces the available numeric daily bounds including its stated uncertainty. It returns an explicit infeasible result instead of violating targets. It is not a clinically authored or validated meal planner.
- An explicitly labeled small illustrative nutrition CSV, FDC search with local fallback, null-preserving nutrient aggregation, GI-derived GL where GI is present, and example recipe decomposition with assumption flags.
- Calibrated-depth geometry utilities, robust non-food plate-plane fitting, height integration, density/packing conversion, confidence-priority mask overlap handling, image quality warnings and safe behavior when models are absent.
- Optional hybrid RF-DETR / SAM 2 / classifier / CLIP / Depth Anything V2 adapter. Its source/API and dependency configuration have been inspected; trained-checkpoint inference has **not** run.
- Dataset checksum/resume downloader, taxonomy/grouped split preparation, COCO export, three-backbone classification and CNN weight training with Drive-resumable checkpoints, two RF-DETR training sizes, calibrated-geometry feature extraction, train-only PCA, XGBoost validation selection against CNN predictions, separate split-conformal calibration, evaluation utilities and a Colab notebook.

## Important unfinished requirements

This is an honest scope boundary, not a production claim:

| Requested capability | Current state |
|---|---|
| Trained food detection / recognition, including unseen regional dishes | Training scripts and optional adapter; no datasets or trained checkpoints acquired, no model accuracy measured |
| Recognition of **any** food | Cannot be guaranteed; CLIP only ranks supplied candidates, scores are uncalibrated, confirmation always required |
| Metric single-image food mass | Geometry works with supplied calibrated metric depth; ordinary relative monocular depth + a known coin/card is insufficient for reliable vertical scale; image API requires manually entered mass |
| Selected XGBoost/CNN connected to image API | Training/selection utilities exist; current image API does not run the winning mass regressor |
| Nutrition5k item-level training | Not fabricated; dish-level measured mass is supported, item-level masks/mass require new annotations |
| Full clinical rule sets for every major condition | 155 draft catalog files; selected condition guidance and a few numeric limits; many conditions intentionally use generic/supervised fallback |
| Complete ICD-10/ICD-11 mappings | Partial category-level ICD-10-CM; ICD-11 remains unmapped; professional coding review required |
| Real USDA cache and complete micronutrients/purines/oxalate | Small illustrative CSV with explicit provenance warnings; live USDA search available; missing data remain null |
| Regional five-meal menus with swaps | Small Indian/international illustrative library; up to three alternatives checked against numeric constraints; fewer or no swaps when constraints prevent them; not clinically certified |
| Complete automatic mixed-dish decomposition | Two illustrative mixture examples only; no hidden-ingredient recognition claim |
| Clinical efficacy, calibrated intervals, production privacy/security | Not established; intervals in manual logging are illustrative, and local SQLite is not encrypted |

See `data/DATASETS.md` and `models/README.md` before starting ML experiments. The application must not be used for patient treatment decisions in its current state.

## Verification and actual output

See `reports/pytest.txt`, `reports/browser-check.txt`, `reports/docker-build.txt`, and `reports/VERIFICATION.md`. The latest full suite reported:

```text
62 passed, 1 warning in 10.62s
```

The warning concerns the Starlette/AnyIO `BlockingPortal` alias. JavaScript syntax, Python compilation, all 155 YAML files and three generated-image smoke checks also passed. The browser walkthrough exercised desktop/mobile onboarding, condition selection, food calculation, persisted history and the planner, with no JavaScript errors or horizontal overflow.

These tests verify software behavior. The synthetic detection/geometry integration uses an explicitly mocked detector. The three smoke images are generated fixtures, not real photographed foods. **Real-model image → detection → measured weight → nutrition accuracy has not been tested.**

Re-run:

```bash
. .venv/bin/activate
python -m pytest -q
python scripts/validate_rules.py
python scripts/smoke.py
python -m compileall -q app ml scripts
```

Optional JavaScript syntax check: `node --check app/static/app.js`.

## Data → training → evaluation

1. Review dataset/model licenses. Acquire a manageable licensed subset; use the downloader with an independently verified checksum. Follow `data/DATASETS.md` to create reviewed taxonomy and JSONL manifests. No fabricated dataset-to-FDC mappings are supplied.
2. Assign one seed-42 group split **before** generating crops, using physical dish IDs shared across views/tasks. Exact duplicates merge groups; conflicting duplicate labels fail loudly. Near-duplicate detection still requires review.
3. Run the Colab notebook `notebooks/train_colab.ipynb` with a 12–16 GB GPU. It saves checkpoints in Drive. The scripts use batch size 1 + accumulation for detectors and mixed precision for CNNs; actual VRAM fit must be measured.
4. Compare RF-DETR Small/Base and ResNet50/EfficientNetV2-S/ConvNeXt-Tiny on the same immutable split. Classifier sampling balances classes only in training. Keep the weight model with lower validation MAE. Train PCA on training embeddings only; calibrate conformal residuals on a separate held-out split.
5. Evaluate once on untouched test data, overall and per class. Supply independently measured calories/carbs for nutrient-error metrics; do not just derive both sides from the same predicted class.

Example commands, after creating the named input artifacts:

```bash
python -m ml.data --manifest data/raw.jsonl --taxonomy data/taxonomy.json --output data/manifest.jsonl
python -m ml.export_coco --manifest data/manifest.jsonl --out data/coco
python -m ml.train_detector --data data/coco --size small --out runs/detector-small
python -m ml.train_detector --data data/coco --size base --out runs/detector-base
python -m ml.train_classifier --manifest data/manifest.jsonl --backbone resnet50 --out runs/resnet50 --resume
python -m ml.train_classifier --manifest data/weight-manifest.jsonl --backbone resnet50 --out runs/cnn-weight --regression --resume
python -m ml.predict_cnn --manifest data/weight-manifest.jsonl --checkpoint runs/cnn-weight/best.pt --output runs/cnn-predictions.jsonl
python -m ml.extract_features --manifest data/weight-manifest.jsonl --checkpoint runs/resnet50/best.pt --output runs/features.npz
python -m ml.train_weight --features runs/features.npz --cnn-predictions runs/cnn-predictions.jsonl --out runs/weight
python -m ml.evaluate --predictions runs/test-predictions.jsonl --out runs/metrics.json --coco-gt data/coco/test/_annotations.coco.json --coco-predictions runs/detector-test.json --model-files runs/detector-small/checkpoint_best_total.pth
```

Repeat the backbone command for `efficientnetv2` and `convnext-tiny`. `scripts/run_experiments.sh` automates detector/classifier/CNN runs after you set its three manifest/path variables. RF-DETR detector resume accepts `--resume` with the actual saved checkpoint path. CNN resume restores optimizer, AMP scaler, random state and epoch. Do not load untrusted `.pt` or Joblib files; they can contain executable serialization.

Feature extraction expects **calibrated geometry** artifacts, and currently uses supplied masks/depth. This is an oracle-geometry experiment, not end-to-end automatic image inference. For a fair automatic-pipeline benchmark, separately compute features from held-out predicted masks and validated depth and label that experiment distinctly.

`ml.evaluate` expects JSONL test predictions with `sample_id`, `split`, `class_name`, `weight_g`, `predicted_g`; optional `kcal`, `predicted_kcal`, `carbs_g`, `predicted_carbs_g`, `low_g`, `high_g`, `latency_ms`. MAPE excludes zero-ground-truth masses and reports the exclusion count. R² is null for constant/singleton groups. Speed reports must record hardware, thread count, warm-up and whether model loading is included.

The six-row blank comparison table is `reports/model_comparison.csv`. No model is declared the winner. RF-DETR Small is a sensible *first experiment* for constrained GPUs, not a measured accuracy/latency recommendation.

## Clinical and uncertainty design

The rule engine uses lower maxima and higher minima when combining conditions. An impossible interval blocks planning. **Clinician-entered targets override defaults**, even if less restrictive; that override is explained. A numeric minimum/maximum alone cannot resolve renal-stage, dialysis, oncology, pregnancy, rare-metabolic or medication decisions. Supervised plans are therefore blocked rather than guessed.

Tier 3 targets are retained in the local profile but not used to generate restrictions. Food-logging nutrient totals remain visible; automatic weight-loss and energy-target features are absent. Food–drug flags use “ask your doctor”; the application never recommends insulin or medication doses. A medication name not recognized by the small string matcher is not proof of no interaction.

BMI is suppressed for children and Tier 3. Adult energy uses Mifflin–St Jeor with selected activity; unspecified/other sex requires a clinician target. It is a maintenance estimate, not a prescribed calorie ceiling. The current app does not generate weight-loss deficits. Pregnancy and pediatric users require clinician plans.

Every logged mass and calculated nutrient has a range. Defaults are ±10% for kitchen-scale entries (composition still varies), at least ±35% for user estimates. These are **illustrative bounds**, not calibrated confidence intervals. Missing nutrient values propagate as unknown. Household assumptions are 240 ml/cup and 150 ml/katori; actual vessels and packing differ. Mixed/hidden foods, oils, soups, stews and layered dishes are particularly uncertain. A reference object helps image scale but cannot eliminate depth ambiguity.

Edit a rule by copying `data/rule-template.yaml` into `app/rules/`, then run the validator. Add the appropriate source, version, review date, reviewer and verified codes. Structural validation does not constitute medical review. All bundled files are drafts with `last_reviewed: null`; none falsely claim a clinician review.

## Privacy

No analytics or third-party frontend assets. SQLite contains unencrypted health information on the local server; protect the host disk, backups and Docker volume. Images are processed in memory and not stored. The optional FDC search sends only the food query to USDA, not the profile. Set `FDC_API_KEY` in your shell before Compose if desired; never commit it. Model downloads occur only through the explicit download script, not silently at inference.

There is no multi-user authentication. The default loopback binding, host allowlist and same-origin write protection are intended for a single trusted user. Do not expose this service publicly without authentication, encryption, access controls and a clinical/privacy review. Deleting the profile cascades to all its logged meals; it does not securely erase disk blocks or external backups.

## Folder structure

```text
nutriscope/
  app/
    main.py               FastAPI endpoints and local-host protections
    schemas.py            Validated profiles, targets and meal inputs
    database.py           SQLite persistence and deletion
    nutrition.py          Nutrient intervals, recipes, FDC search
    rules_engine.py       Tiers, conflicts, clinician overrides, alerts
    planner.py            Constrained illustrative menu planner
    volume.py             Calibrated geometry and overlap resolution
    vision.py             Quality checks and optional hybrid adapter
    rules/                155 versioned draft YAML files
    static/               Buildless HTML, CSS, JavaScript UI
  data/
    foods.csv             Explicitly illustrative food composition data
    recipes.json          Flagged example ingredient ratios
    rule-template.yaml    New-condition template
    DATASETS.md           Provenance, licenses and manifest contract
    samples/              Three generated smoke fixtures
  ml/
    data.py               Duplicate-aware grouped splits and taxonomy
    export_coco.py        RF-DETR COCO export
    train_detector.py     RF-DETR Small/Base training
    train_classifier.py   Three classifiers and CNN mass baseline
    predict_cnn.py        Paired CNN predictions
    extract_features.py   Geometry features and backbone embeddings
    train_weight.py       PCA/XGBoost, model selection, conformal radius
    evaluate.py           Detection/weight/nutrient metrics
  notebooks/train_colab.ipynb
  scripts/                Download, inference, checks, smoke, experiments
  tests/                  Unit, API, clinical-tier and contract integration
  models/README.md        Checkpoint contract and known integration gaps
  reports/               Actual logs, screenshots and blank metrics table
  config.yaml
  requirements.txt
  requirements.lock
  requirements-ml.txt
  requirements-ml.lock
  Dockerfile
  compose.yaml
```
