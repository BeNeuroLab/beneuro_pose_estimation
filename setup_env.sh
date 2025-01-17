#!/bin/bash

# Automatically exit if a command fails
set -e

# Create Conda environment with required channels and packages
conda create -y -n bnp -c conda-forge -c nvidia -c sleap -c anaconda sleap typer rich

# Conda init
conda init

# Activate the Conda environment
conda activate bnp

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