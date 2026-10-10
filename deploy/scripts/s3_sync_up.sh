#!/usr/bin/env bash
# Push generated/durable workspace prefixes back up to S3 after a run. No-op if BC_S3_BUCKET unset.
# Deliberately excludes raw/ (input dataset truth — a compute container should never write it back,
# only ever read it) and the per-container .tmp scratch dir.
set -euo pipefail

: "${BC_HOME:=/app}"
WS="$(PYTHONPATH="${BC_HOME}/code/src" python3 -c "import config as C; print(C.WS)")"

if [ -z "${BC_S3_BUCKET:-}" ]; then
  echo "[s3_sync_up] BC_S3_BUCKET not set — skipping, results stay local only (${WS})"
  exit 0
fi

PREFIX="${BC_S3_PREFIX:-is360-breast-poc}"
DIRS=(processed research models results reports audit manifests logs)

for d in "${DIRS[@]}"; do
  src="${WS}/${d}/"
  dst="s3://${BC_S3_BUCKET}/${PREFIX}/${d}/"
  if [ -d "$src" ]; then
    echo "[s3_sync_up] ${src} -> ${dst}"
    # Governance note: manifests/ (ms_access_log.csv, the append-only MS-access ledger) should sit in
    # a bucket/prefix with S3 Versioning + Object Lock (governance-mode) enabled at the bucket-policy
    # level — that is an account/IAM decision outside this script's scope, not something `aws s3 sync`
    # enforces; see deploy/README_DEPLOY.md.
    aws s3 sync "$src" "$dst" --only-show-errors
  fi
done

echo "[s3_sync_up] done"
