#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"          # lavora sempre dalla cartella dello script
mkdir -p results/fifa

sudo docker run -d --name xgb_fifa --gpus '"device=0"' \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/config:/app/config:ro" \
  -v "$PWD/results:/app/results" \
  xgb-xai-stability:1.0 \
  --strategy optuna --runs 1000 --seed 42 \
  --config config/config.json \
  --train data/fifa/train_fifa.csv \
  --test data/fifa/test_fifa.csv \
  --sep comma \
  --output-dir results/fifa