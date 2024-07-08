#!/bin/sh

echo "::group::Setup Repolinter"
set -euo

echo "[INFO] Installing todogroup/repolinter"
npm install -g log-symbols 
npm install -g repolinter 

echo "[INFO] Executing:"
echo "[INFO] repolinter $*"
echo "::endgroup::"

sh -c "repolinter lint ."