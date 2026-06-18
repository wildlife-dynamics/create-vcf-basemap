#!/bin/bash
set -euo pipefail

RECIPES=("release/vcf-basemap")

export HATCH_VCS_VERSION=$(hatch version)
echo "HATCH_VCS_VERSION=$HATCH_VCS_VERSION"

rm -rf /tmp/ecoscope-workflows-custom/release/artifacts
mkdir -p /tmp/ecoscope-workflows-custom/release/artifacts

for rec in "${RECIPES[@]}"; do
    echo "Building $rec"
    rattler-build build \
      --recipe "$(pwd)/publish/recipes/${rec}.yaml" \
      --output-dir /tmp/ecoscope-workflows-custom/release/artifacts \
      --channel https://prefix.dev/ecoscope-workflows \
      --channel conda-forge
done
