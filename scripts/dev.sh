#!/usr/bin/env bash
set -euo pipefail
python app/index/build_index.py --src data --out app/index/store
uvicorn app.main:app --reload --port 8000
