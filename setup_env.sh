#!/bin/bash

# Python
conda create -y -n bnp python=3.7

# Create Conda environment with required channels and packages
conda create -y -n bnp -c conda-forge -c nvidia -c sleap -c anaconda sleap

# Activate the Conda environment
conda activate bnp

# Extra packages
conda install -y matplotlib numpy pandas seaborn typer rich pip

# Remove opencv pypi version to avoid conflicts
pip uninstall -y opencv-python-headless

# Install the required version of OpenCV
pip install "opencv-contrib-python<4.7.0"

# Install sleap_anipose and the required version of anipose
pip install sleap_anipose
pip install "anipose<1.1"

# Upgrade apptools to the latest version
pip install --upgrade apptools

# Install package in editable form
pip install -e ./

# Print success message
echo "Environment setup complete!"