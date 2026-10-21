# Verification record — 21 October 2026

This record separates executed software checks from unexecuted ML/clinical work.

## Executed

- Python **3.12.14** virtual environment; direct and transitive application dependencies pinned in `requirements.lock`.
- `python -m pytest -q`: **62 passed**, one Starlette/AnyIO deprecation warning. Actual output is in `pytest.txt`.
- Rule validator: **155 structurally valid YAML rules**. All remain draft; zero clinical-review claims.
- Three generated image fixture smoke checks: **3/3 passed**. No fabricated detections when model weights are absent; dark/low-detail/missing-reference warnings exercised.
- Python compile and JavaScript syntax checks passed.
- Desktop (1440 px) and mobile (390 px) Chromium walkthrough: profile wizard → searchable condition/allergy entry → guidance → meal calculation → journal persistence → plan request. No uncaught JavaScript errors, no mobile horizontal overflow. See `browser-check.txt` and PNG screenshots.
- Docker image built from Python 3.12.14 slim with the complete application lockfile. See `docker-build.txt`.
- Docker runtime verified using a non-root user, read-only root filesystem, temporary writable SQLite storage, dropped Linux capabilities and loopback-bound port. See `docker-runtime.txt`.
- ML dependency graph resolved into `requirements-ml.lock`; optional GPU packages were not installed or executed. Source inspection confirmed RF-DETR Small/Base APIs. Transformers 4.51.3 and 4.55.4 did not include SAM 2; the pin was corrected to **4.56.2** after inspecting the wheel. The adapter uses SAM 2.1 Tiny and single-mask output.
- Hugging Face repository metadata was queried to verify model repositories and pin immutable revision SHAs in `data/model-sources.json`; model weights were not downloaded.

## Fixes encountered during verification

1. `/home/agent/.cache/uv` was read-only; used `/tmp/nutri-uv`.
2. Sandbox networking initially blocked package downloads and local socket creation; used scoped network access. In-process API testing uses socket-backed event-loop internals.
3. Docker Buildx attempted to write a read-only home directory; moved its cache/config to `/tmp/nutriscope-buildx`. Preserved Docker credentials/proxy settings and mounted the system CA bundle as a build secret.
4. Browser test's string-evaluation wait was blocked by the app's strict Content Security Policy; changed the test to wait on a DOM locator, retaining the CSP.
5. ML source review found an unsupported SAM 2 import in the original Transformers pin; corrected pin and model repository.
6. Detector training was configured with `run_test=False` so model selection does not repeatedly inspect the test set. Best classifier checkpoints now omit optimizer/RNG objects, allowing weights-only inference loading; full resume checkpoints retain them.

## Not executed / not established

- No RF-DETR fine-tuning, classifier training, SAM 2 inference, Depth Anything inference, XGBoost training, CNN mass training, CLIP calibration, conformal calibration, CPU model latency or end-to-end food accuracy benchmark.
- The image → detections → geometry → mass → nutrients → alerts integration test uses **explicitly synthetic/mock detections** and calibrated synthetic depth. This is contract verification, not model validation.
- The hemisphere geometry test verifies numerical integration under the documented near-overhead metric-depth assumptions, not single-photo metric scale recovery.
- No real-food sample smoke test. Smoke images are generated empty/dark/illustrative scenes.
- No licensed research datasets acquired, no trained checkpoint winner chosen, no model accuracy metrics populated.
- No clinical/dietitian review, complete ICD mapping, validated nutrition composition cache or medical efficacy validation.
- No public deployment or external transmission of patient data.

Use the README's limitations table when assessing suitability. Passing software tests does not make the prototype a clinical product.
