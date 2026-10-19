#!/usr/bin/env bash
# Runs the same steps as .github/workflows/ci.yml on a local machine.
# Exits non-zero on the first failing step (set -e), like the CI pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "Lint (flake8)";            $PY -m flake8 src api tests; echo "flake8: no issues"
step "Unit tests (pytest)";      $PY -m pytest -v --junitxml=test-results.xml
step "Download & clean data";    $PY -m src.data
step "EDA figures";              $PY -m src.eda
step "Train & track (MLflow)";   $PY -m src.train 2>&1 | grep -v -i warning | tee train.log
step "Docker build";             docker build -q -t heart-disease-api:ci .
step "Container smoke test"
docker rm -f heart-ci >/dev/null 2>&1 || true
docker run -d --name heart-ci -p 8099:8000 heart-disease-api:ci >/dev/null
for i in $(seq 1 30); do curl -sf localhost:8099/health >/dev/null && break; sleep 1; done
curl -sf localhost:8099/health; echo
curl -sf -X POST localhost:8099/predict -H 'Content-Type: application/json' -d @sample_request.json | tee predict.json; echo
grep -q '"confidence"' predict.json
docker rm -f heart-ci >/dev/null
step "PIPELINE PASSED"
