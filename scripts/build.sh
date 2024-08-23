#!/bin/sh

set -ev

python3 -m pip install --upgrade pip build

BASEDIR=$(dirname "$(readlink -f "$(dirname $0)")")
DISTDIR=dist



cd $BASEDIR
echo "building"
mkdir -p $DISTDIR
rm -rf ${DISTDIR:?}/*
python3 -m build --outdir "$BASEDIR/dist/"

echo "build complete"