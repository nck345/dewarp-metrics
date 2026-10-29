#!/usr/bin/env bash
# Compile SIFT Flow MEX files on Linux / Google Colab for GNU Octave
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEX_DENSE="$SCRIPT_DIR/matlab/SIFTflow/mexDenseSIFT"
MEX_FLOW="$SCRIPT_DIR/matlab/SIFTflow/mexDiscreteFlow"

echo "=== Compiling SIFT Flow MEX for GNU Octave ==="

echo "1/2 Compiling mexDenseSIFT..."
cd "$MEX_DENSE"
mkoctfile --mex mexDenseSIFT.cpp Matrix.cpp Vector.cpp -o mexDenseSIFT.mex

echo "2/2 Compiling mexDiscreteFlow..."
cd "$MEX_FLOW"
mkoctfile --mex mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp -o mexDiscreteFlow.mex

cd "$SCRIPT_DIR"
echo "✅ Compilation completed successfully! MEX files are ready in matlab/SIFTflow/"
