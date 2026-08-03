#!/usr/bin/env bash
set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SIM_ROOT/../.." && pwd)"
SUBJECT_ID="${1:-1}"

python3 "$REPO_ROOT/src/public_datasets/export_limb_heldout_models.py" \
  --subject "$SUBJECT_ID" \
  --force
