#!/bin/bash
# Rebuild MXtalTools Sphinx documentation
set -e

cd "$(dirname "$0")"

# Clean previous build
rm -rf _build source/_autosummary

# Build HTML docs
python -m sphinx build -b html . _build/html

echo ""
echo "Build complete. Open _build/html/index.html to view."
