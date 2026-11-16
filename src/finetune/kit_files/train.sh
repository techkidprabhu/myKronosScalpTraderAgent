#!/usr/bin/env bash
# Fine-tune Kronos on this computer's NVIDIA GPU. Sets Python up the first
# time (in ~/.kronos-trainer, shared by every kit you export), then trains.
#   ./train.sh              ./train.sh --steps 6000 --batch 16
set -e
cd "$(dirname "$0")"
HOME_DIR="${KRONOS_TRAINER_HOME:-$HOME/.kronos-trainer}"
VENV="$HOME_DIR/venv"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "First run: setting up Python packages in $VENV (a few GB, several minutes)..."
  mkdir -p "$HOME_DIR"
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install torch --index-url "$TORCH_INDEX"
fi
"$VENV/bin/python" -m pip install -q -r requirements.txt
exec "$VENV/bin/python" train_kronos.py "$@"
