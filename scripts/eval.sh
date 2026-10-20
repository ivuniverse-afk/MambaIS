#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/scannetv2.yaml}"
CHECKPOINT="${2:-outputs/scannetv2/last.pt}"
python tools/evaluate.py --config "$CONFIG" --checkpoint "$CHECKPOINT"

