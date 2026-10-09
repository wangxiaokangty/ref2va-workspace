#!/usr/bin/env bash
# References in the manifest are relative to this directory, not the manifest file.
set -euo pipefail
: "${H3_TRAINER_REPO:?Set H3_TRAINER_REPO to an absolute upstream checkout path}"
: "${H3_MODEL_PATH:?Set H3_MODEL_PATH to the diffusers-layout model root}"
sample_root="$(cd "$(dirname "$0")" && pwd)"
sample_name="${1:-01_single_image}"
case "$sample_name" in
  01_single_image|02_two_images|03_image_audio|04_image_video|05_video_audio) ;;
  *) echo "Unknown sample: $sample_name" >&2; exit 2 ;;
esac
cd "$sample_root"
python "$H3_TRAINER_REPO/scripts/process_dataset.py" "samples/$sample_name.json" \
  --model-path "$H3_MODEL_PATH" --references \
  --resolution-bucket 832x480x124 --reference-canvas native \
  --output "$sample_root/.precomputed/$sample_name" --decode 1
