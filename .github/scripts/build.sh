#!/bin/sh

set -ev

# Get the latest versions of packaging tools
python3 -m pip install --upgrade pip build setuptools wheel

python3 -m build