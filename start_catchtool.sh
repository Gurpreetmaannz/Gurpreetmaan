#!/usr/bin/env bash
# Open catchtool in your browser. The first run sets itself up (a few minutes).
set -e
cd "$(dirname "$0")"
command -v conda >/dev/null || { echo "Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q '^catchtool ' || conda create -y -n catchtool python=3.12 -c conda-forge rasterio geopandas numba matplotlib
conda activate catchtool
pip install -q -e .
catchtool-gui
