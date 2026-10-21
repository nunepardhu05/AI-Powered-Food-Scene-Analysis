# Models are not bundled

Manual logging works with no ML packages or model weights. The default image endpoint returns `models_unavailable`; it does not claim no food is present.

Optional model integration is implemented but has not been exercised with downloaded weights. Use Python 3.12, `requirements-ml.txt`, and the notebook. Artifact contracts:

- `detector.pth`: trained RF-DETR checkpoint, matching model size and canonical class IDs.
- `classifier.pt`: classifier `best.pt` produced by the training script, matching `labels` order.
- `sam2/`, `clip/`, `depth/`: local Hugging Face saved models and processors from `scripts/fetch_models.py`.
- `manifest.json`: JSON with `detector_size` (`small` or `base`), `backbone` (`resnet50`, `tf_efficientnetv2_s` or `convnext_tiny`), and `labels` (ordered canonical names). Optional `open_vocabulary` adds extra CLIP candidate labels.

Check each upstream model license. `data/model-sources.json` pins verified Hugging Face repository revisions; the download script uses them explicitly. SAM 2.1 Tiny is used through Transformers 4.56.2; earlier pins lacked the required API. SAM2, CLIP and Depth Anything V2 use their original authors' terms; do not assume RF-DETR's package license covers all pretrained weight uses. The application does no automatic model download during inference.

The adapter fuses detector, classifier and CLIP candidate scores 0.25/0.45/0.30. These are uncalibrated ranking scores, not validated confidence estimates. CLIP ranks the supplied vocabulary; it cannot recognize every possible food. Confirmation is always requested. SAM2 overlaps go to the higher detector confidence, and both affected items are flagged. The overlay uses sampled mask pixels for transport; geometry computation must use full-resolution masks.

Depth Anything V2 Small supplies relative depth. The adapter deliberately does not turn it into a metric volume or weight. A coin/card/plate gives a lateral image scale, but not the unknown depth scale and offset. To report a metric volume, provide calibrated camera-axis depth in cm and near-overhead geometry via the `app.volume` API. Depth alignment, intrinsics, calibration accuracy, lens distortion, occlusion and plate tilt need validation. A calibrated weight-regression artifact can eventually replace manual mass; the current HTTP image adapter does not load the selected weight model.

The CPU Docker image does not install the large optional ML stack. For CPU model experiments install the ML requirements in a separate environment and run uvicorn there, or build a derived image with the reviewed ML dependency lock. Memory and inference latency are not measured yet.
