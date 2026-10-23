#!/bin/bash
# Rebuild MXtalTools Sphinx documentation.
# Needs torch and torch_geometric installed (CPU builds suffice); see
# docs/requirements-torch.txt and docs/requirements.txt.
set -e

cd "$(dirname "$0")"

# Clean previous build
rm -rf _build source/_autosummary

# Build HTML docs
python -m sphinx build -b html . _build/html

echo ""
echo "Build complete. Open _build/html/index.html to view."
