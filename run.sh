#!/usr/bin/env bash
# One entry point. Usage: ./run.sh setup | test | mutation | bench [args] | try "sentence" | serve
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
case "${1:-serve}" in
  setup)    uv venv --python 3.12 .venv && uv pip install --python $PY -r requirements.txt ;;
  test)     $PY -m pytest -q tests ;;
  mutation) $PY scripts/mutation_check.py ;;
  bench)    shift; $PY -m bench.run "$@" ;;
  try)      shift; $PY -m structreq "$@" ;;
  serve)    shift || true; $PY -m structreq serve "$@" ;;
  *)        echo "usage: ./run.sh setup|test|mutation|bench|try|serve"; exit 1 ;;
esac
