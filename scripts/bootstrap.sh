#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# End-to-end smoke test of the whole pipeline:
#   build -> test -> self-play export -> train -> export weights -> play
#
# Runs the small `dev` model by default so it finishes in a couple of minutes
# even without a GPU. Pass MODEL=s once ROCm is working to use the spec-sized
# network.
#
# Usage:
#   ./scripts/bootstrap.sh
#   MODEL=s STEPS=2000 DEVICE=cuda ./scripts/bootstrap.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-dev}"
STEPS="${STEPS:-300}"
GAMES="${GAMES:-10}"
PIECES="${PIECES:-120}"
SIMS="${SIMS:-16}"
DEVICE="${DEVICE:-cpu}"
PYTHON="${PYTHON:-python3}"

mkdir -p data models

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "1/6  Building the engine"
make -s test

say "2/6  Building the tools"
make -s tools

say "3/6  Generating a training set ($GAMES games)"
./build/tetra_cli export data/train.tetradat "$GAMES" "$PIECES" "$SIMS"

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    cat <<'EOF'

PyTorch is not installed, so the training steps are being skipped.
The engine side of the pipeline is verified and data/train.tetradat is ready.

To train, see docs/SETUP.md. On an AMD GPU, in short:

    python -m venv .venv && source .venv/bin/activate
    pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
    pip install -r trainer/requirements.txt

EOF
    exit 0
fi

say "4/6  Training ($MODEL model, $STEPS steps, device=$DEVICE)"
"$PYTHON" trainer/train.py data/train.tetradat \
    --steps "$STEPS" --model "$MODEL" --device "$DEVICE" --save models/gen1.pt

say "5/6  Exporting weights for the C++ engine"
"$PYTHON" trainer/export_weights.py models/gen1.pt models/gen1.tetrawts

say "6/6  Playing with the trained network"
./build/tetra_cli play models/gen1.tetrawts "$PIECES" "$SIMS"

cat <<'EOF'

Pipeline complete.

  data/train.tetradat    the training set
  models/gen1.pt         the PyTorch checkpoint
  models/gen1.tetrawts   weights the C++ engine can load

Next: docs/SETUP.md section 4 describes the iterative loop.
EOF