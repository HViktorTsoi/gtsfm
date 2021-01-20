#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previously installed using conda
##########################################################

echo "Running .github/scripts/python.sh..."
conda init
conda info --envs

wget -O 2020_01_13_gtsam_python38_wheel.zip --no-check-certificate "https://drive.google.com/uc?export=download&id=1b7zoYopU7jN3D62fuZMqwQgZdhZ4cH6P"
unzip 2020_01_13_gtsam_python38_wheel.zip
pip install gtsam-4.1.1-cp38-cp38-manylinux2014_x86_64.whl

##########################################################
# Install GTSFM as a module
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .

##########################################################
# Run GTSFM unit tests
##########################################################

cd $GITHUB_WORKSPACE/tests
python -m unittest discover
