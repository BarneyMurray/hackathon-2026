#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
cd data
if [ -d imagenette2-320 ]; then
  echo "imagenette2-320 already present, skipping download"
  exit 0
fi
curl -L -o imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
rm imagenette2-320.tgz
echo "done: $(find imagenette2-320 -type f | wc -l) files"
