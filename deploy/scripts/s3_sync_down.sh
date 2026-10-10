#!/usr/bin/env bash
# Pull the durable workspace prefixes from S3 down into the local workspace before a run.
# No-op (and exits 0) if BC_S3_BUCKET is unset, so local-only / DEV-only use (as validated this
# session) keeps working unchanged without S3 configured.
#
# Layout mirrors config.py's DIRS exactly (config.py's own header: "the workspace mirrors the S3
# bucket layout of the IS360 Breast Imaging Platform architecture"):
#   raw/tenant=is360/source=mias, processed/tenant=is360/source=mias, research/tenant=is360,
#   models/tenant=is360, results/tenant=is360, reports/tenant=is360, audit, manifests, logs
set -euo pipefail

: "${BC_HOME:=/app}"
# Ask config.py itself for the workspace root rather than re-deriving the path logic here.
WS="$(PYTHONPATH="${BC_HOME}/code/src" python3 -c "import config as C; print(C.WS)")"

if [ -z "${BC_S3_BUCKET:-}" ]; then
  echo "[s3_sync_down] BC_S3_BUCKET not set — skipping, using local workspace only (${WS})"
  exit 0
fi

PREFIX="${BC_S3_PREFIX:-is360-breast-poc}"
DIRS=(raw processed research models results reports audit manifests logs)

for d in "${DIRS[@]}"; do
  src="s3://${BC_S3_BUCKET}/${PREFIX}/${d}/"
  dst="${WS}/${d}/"
  mkdir -p "$dst"
  echo "[s3_sync_down] ${src} -> ${dst}"
  aws s3 sync "$src" "$dst" --only-show-errors || echo "[s3_sync_down] WARN: sync failed for ${d} (prefix may not exist yet — fine on first run)"
done

echo "[s3_sync_down] done"
