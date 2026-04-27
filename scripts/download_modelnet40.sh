#!/usr/bin/env bash
# Download and lay out ModelNet40 in the directory structure expected by
# Graphite's ``ModelNetDataset``.
#
#     <MODELNET_DIR>/
#         object_names.txt          # 40 category names, one per line
#         airplane/{train,test}/*.off
#         bathtub/ {train,test}/*.off
#         …
#
# Usage:
#     ./scripts/download_modelnet40.sh
#     MODELNET_DIR=/data/modelnet40 ./scripts/download_modelnet40.sh
set -euo pipefail

MODELNET_DIR=${MODELNET_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/data/modelnet/"}
mkdir -p "$MODELNET_DIR"
cd "$MODELNET_DIR"

URL="http://modelnet.cs.princeton.edu/ModelNet40.zip"
ZIP="ModelNet40.zip"

if [ ! -f "$ZIP" ]; then
    echo "Downloading ModelNet40 (~450 MB) ..."
    curl -L --retry 3 -o "$ZIP" "$URL"
fi

if [ ! -d "ModelNet40" ]; then
    echo "Extracting ..."
    unzip -q "$ZIP"
fi

shopt -s dotglob nullglob
mv ModelNet40/* . 2>/dev/null || true
rmdir ModelNet40 2>/dev/null || true
rm -rf __MACOSX 2>/dev/null || true

echo "Writing object_names.txt ..."
ls -d */ 2>/dev/null \
    | sed 's:/$::' \
    | grep -v '^processed$' \
    | sort > object_names.txt

count=$(wc -l < object_names.txt)
echo "Found $count categories in $MODELNET_DIR"
if [ "$count" -lt 40 ]; then
    echo "WARNING: expected 40 categories, found $count" >&2
fi
