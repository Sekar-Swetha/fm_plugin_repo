#!/usr/bin/env bash
# One-shot supervisor handover: install deps, run unit tests, run verification
# on every MotionEditor case folder, and print a one-page summary.
#
# Usage from the repo root (where MotionEditor/ and mediapipe_motioneditor_plugin/ live):
#   bash mediapipe_motioneditor_plugin/run_all_proofs.sh [path/to/MotionEditor/data]
#
# Default MotionEditor data path: motionEditor/MotionEditor/data
#
# Produces:
#   mediapipe_motioneditor_plugin/proofs/case-X/{side_by_side,overlay,diff,report.{json,md}}
#   and a top-level summary printed to stdout.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-${HERE}/../motionEditor/MotionEditor/data}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root not found: $DATA_ROOT" >&2
  echo "Pass the MotionEditor data directory as the first argument." >&2
  exit 1
fi

echo "==> Installing requirements"
pip install -q -r "${HERE}/requirements.txt"

echo
echo "==> Running offline unit tests"
( cd "$HERE" && python -m unittest discover -s tests -v )

echo
echo "==> Running verification on each MotionEditor case folder"
mkdir -p "${HERE}/proofs"

# A case folder is expected to contain `images/` and (optionally)
# `source_condition/openposefull/`. Iterate over case-* subdirs.
for case_dir in "$DATA_ROOT"/case-*; do
  [[ -d "$case_dir" ]] || continue
  case_name="$(basename "$case_dir")"
  frames_dir="$case_dir/images"
  ref_dir="$case_dir/source_condition/openposefull"
  out_dir="${HERE}/proofs/${case_name}"
  echo
  echo "--- ${case_name} ---"
  if [[ ! -d "$frames_dir" ]]; then
    echo "  no images/ subfolder, skipping"
    continue
  fi
  ref_arg=()
  if [[ -d "$ref_dir" ]]; then
    ref_arg=(--openpose-ref "$ref_dir")
  fi
  python "${HERE}/verify.py" \
    --frames "$frames_dir" \
    --out "$out_dir" \
    "${ref_arg[@]}"
  echo "  -> proofs in: $out_dir"
done

echo
echo "==> Done. Per-case reports:"
find "${HERE}/proofs" -name 'report.md' -print
