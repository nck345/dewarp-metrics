#!/usr/bin/env bash
# Compile SIFT Flow MEX files on Linux / Google Colab for GNU Octave
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEX_DENSE="$SCRIPT_DIR/matlab/SIFTflow/mexDenseSIFT"
MEX_FLOW="$SCRIPT_DIR/matlab/SIFTflow/mexDiscreteFlow"

echo "=== Compiling SIFT Flow MEX for GNU Octave ==="

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null; then
    SUDO="sudo"
fi

# Clean up broken repositories like r2u if present to avoid apt-get update failure
if [ -d "/etc/apt/sources.list.d" ]; then
    $SUDO rm -f /etc/apt/sources.list.d/*r2u* 2>/dev/null || true
fi

if ! command -v mkoctfile &> /dev/null; then
    echo "Installing octave development package (octave-dev) for mkoctfile..."
    $SUDO apt-get update -qq || true
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends octave octave-dev octave-image -qq || \
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y octave octave-dev -qq || \
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y octave-dev -qq || \
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y liboctave-dev -qq || true
fi

if ! command -v mkoctfile &> /dev/null; then
    for cand in /usr/bin/mkoctfile /usr/local/bin/mkoctfile /usr/lib/octave/*/mkoctfile; do
        if [ -x "$cand" ]; then
            export PATH="$(dirname "$cand"):$PATH"
            break
        fi
    done
fi

if ! command -v mkoctfile &> /dev/null; then
    echo "❌ Error: mkoctfile not found! Please install octave-dev."
    exit 1
fi

echo "1/2 Compiling mexDenseSIFT..."
cd "$MEX_DENSE"
if ! mkoctfile -O3 -march=native -funroll-loops --mex -fpermissive mexDenseSIFT.cpp Matrix.cpp Vector.cpp -o mexDenseSIFT.mex; then
    echo "Retrying mexDenseSIFT without -march=native..."
    mkoctfile -O3 -funroll-loops --mex -fpermissive mexDenseSIFT.cpp Matrix.cpp Vector.cpp -o mexDenseSIFT.mex
fi
cp -f mexDenseSIFT.* "$SCRIPT_DIR/matlab/SIFTflow/" 2>/dev/null || true
cp -f mexDenseSIFT.* "$SCRIPT_DIR/matlab/" 2>/dev/null || true

echo "2/2 Compiling mexDiscreteFlow..."
cd "$MEX_FLOW"
if ! mkoctfile -O3 -march=native -funroll-loops --mex -fpermissive mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp -o mexDiscreteFlow.mex; then
    echo "Retrying mexDiscreteFlow without -march=native..."
    mkoctfile -O3 -funroll-loops --mex -fpermissive mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp -o mexDiscreteFlow.mex
fi
cp -f mexDiscreteFlow.* "$SCRIPT_DIR/matlab/SIFTflow/" 2>/dev/null || true
cp -f mexDiscreteFlow.* "$SCRIPT_DIR/matlab/" 2>/dev/null || true

cd "$SCRIPT_DIR"

if command -v octave &> /dev/null; then
    echo "Verifying MEX binaries and multi-channel imfilter in GNU Octave..."
    octave --no-gui --silent --eval "try, pkg load image; catch, end; addpath('$SCRIPT_DIR/matlab', '-begin'); addpath('$SCRIPT_DIR/matlab/SIFTflow', '-begin'); addpath('$MEX_DENSE', '-begin'); addpath('$MEX_FLOW', '-begin'); s = mexDenseSIFT(zeros(10,10,'double'), 3, 1); test_f = imfilter(zeros(4,4,128,'uint8'), [1 2 1; 2 4 2; 1 2 1]/16, 'same', 'replicate'); disp('✅ Octave runtime verification successful: mexDenseSIFT and multi-channel imfilter are working!');"
fi

echo "✅ Compilation completed successfully! MEX files are ready in matlab/SIFTflow/"
