#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/scannetv2.yaml}"
OUTPUT="${2:-outputs/scannetv2}"
python tools/train.py --config "$CONFIG" --output "$OUTPUT"

