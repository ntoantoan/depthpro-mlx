#!/usr/bin/env bash
# Fetch the official PyTorch implementation and checkpoint (needed only for
# conversion, parity tests and baseline benchmarks).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d third_party/ml-depth-pro ]; then
  git clone --depth 1 https://github.com/apple/ml-depth-pro third_party/ml-depth-pro
fi
mkdir -p checkpoints
if [ ! -f checkpoints/depth_pro.pt ]; then
  curl -L --fail -o checkpoints/depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
fi
echo "reference code and checkpoint ready"
