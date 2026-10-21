#!/usr/bin/env bash
set -euo pipefail
python scripts/validate_rules.py
python -m pytest -q
python scripts/smoke.py
python -m compileall -q app ml scripts
node --check app/static/app.js
