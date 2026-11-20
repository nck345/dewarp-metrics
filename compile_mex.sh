#!/usr/bin/env bash
# Compile SIFT Flow MEX files on Linux / Google Colab for GNU Octave
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEX_DENSE="$SCRIPT_DIR/matlab/SIFTflow/mexDenseSIFT"
MEX_FLOW="$SCRIPT_DIR/matlab/SIFTflow/mexDiscreteFlow"

echo "=== Compiling SIFT Flow MEX for GNU Octave ==="

if ! command -v mkoctfile &> /dev/null; then
    echo "Installing octave development packages (octave-dev liboctave-dev) for mkoctfile..."
    apt-get update -qq && apt-get install -y octave-dev liboctave-dev -qq || true
fi

if ! command -v mkoctfile &> /dev/null; then
    echo "❌ Error: mkoctfile not found! Please install octave-dev or liboctave-dev."
    exit 1
fi

echo "1/2 Compiling mexDenseSIFT..."
cd "$MEX_DENSE"
mkoctfile -O3 --mex -fpermissive mexDenseSIFT.cpp Matrix.cpp Vector.cpp -o mexDenseSIFT.mex
cp -f mexDenseSIFT.mex "$SCRIPT_DIR/matlab/SIFTflow/" 2>/dev/null || true
cp -f mexDenseSIFT.mex "$SCRIPT_DIR/matlab/" 2>/dev/null || true

echo "2/2 Compiling mexDiscreteFlow..."
cd "$MEX_FLOW"
mkoctfile -O3 --mex -fpermissive mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp -o mexDiscreteFlow.mex
cp -f mexDiscreteFlow.mex "$SCRIPT_DIR/matlab/SIFTflow/" 2>/dev/null || true
cp -f mexDiscreteFlow.mex "$SCRIPT_DIR/matlab/" 2>/dev/null || true

cd "$SCRIPT_DIR"

if command -v octave &> /dev/null; then
    echo "Verifying MEX binaries in GNU Octave..."
    octave --no-gui --silent --eval "addpath('$SCRIPT_DIR/matlab'); addpath('$SCRIPT_DIR/matlab/SIFTflow'); addpath('$MEX_DENSE'); addpath('$MEX_FLOW'); s = mexDenseSIFT(zeros(10,10,'double'), 3, 1); disp('✅ Octave runtime verification successful: mexDenseSIFT is working!');"
fi

echo "✅ Compilation completed successfully! MEX files are ready in matlab/SIFTflow/"

