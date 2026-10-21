#!/usr/bin/env bash
set -euo pipefail
: "${MANIFEST:?Set MANIFEST to the prepared classification JSONL}"
: "${COCO_DIR:?Set COCO_DIR to exported COCO splits}"
: "${WEIGHT_MANIFEST:?Set WEIGHT_MANIFEST to measured dish-level regression JSONL}"
EXPERIMENT_OUT="${EXPERIMENT_OUT:-/content/drive/MyDrive/nutriscope-runs}"
for size in small base; do
  python -m ml.train_detector --data "$COCO_DIR" --size "$size" --out "$EXPERIMENT_OUT/detector-$size" --epochs 30
done
for backbone in resnet50 efficientnetv2 convnext-tiny; do
  python -m ml.train_classifier --manifest "$MANIFEST" --backbone "$backbone" --out "$EXPERIMENT_OUT/classifier-$backbone" --epochs 20 --resume
  python -m ml.train_classifier --manifest "$WEIGHT_MANIFEST" --backbone "$backbone" --out "$EXPERIMENT_OUT/cnn-weight-$backbone" --epochs 20 --regression --resume
  python -m ml.predict_cnn --manifest "$WEIGHT_MANIFEST" --checkpoint "$EXPERIMENT_OUT/cnn-weight-$backbone/best.pt" --output "$EXPERIMENT_OUT/cnn-$backbone.jsonl"
done
