#!/usr/bin/env bash
set -e

echo "[1/3] Run pipeline..."
python -m app run --input data --out outputs --config configs/default.yaml

echo "[2/3] Start API on 8001..."
echo "Open: http://127.0.0.1:8001/docs"
uvicorn app.api:app --host 0.0.0.0 --port 8001

