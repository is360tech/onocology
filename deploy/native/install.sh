#!/usr/bin/env bash
# No Docker. Two plain venvs: one thin (API), one full (bc.py pipeline). Idempotent — safe to
# re-run on every deploy (the post-receive hook calls this after every `git push`).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[install] api venv"
python3 -m venv "$ROOT/.venv-api"
"$ROOT/.venv-api/bin/pip" install --quiet --upgrade pip
"$ROOT/.venv-api/bin/pip" install --quiet -r deploy/requirements-api.txt

echo "[install] pipeline venv (versions pinned to match provenance.py's recorded stack)"
python3 -m venv "$ROOT/.venv-pipeline"
"$ROOT/.venv-pipeline/bin/pip" install --quiet --upgrade pip
"$ROOT/.venv-pipeline/bin/pip" install --quiet \
    numpy==2.5.3 scipy==1.18.1 pandas==3.0.5 scikit-learn==1.9.1 \
    scikit-image==0.26.0 pillow==12.3.0 joblib==1.6.0 matplotlib==3.11.2 pyyaml==6.0.2

echo "[install] done: $ROOT/.venv-api, $ROOT/.venv-pipeline"
