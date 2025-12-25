#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "== Repo root =="
pwd

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "== Creating venv (.venv) =="
  python -m venv .venv
fi

# Activate venv (mac/linux/git-bash). If you're using Windows CMD/PowerShell, use run_final.ps1 instead.
source .venv/bin/activate

echo "== Installing requirements =="
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "== Stage B dataset build (Script 10) =="
python scripts/10_build_stage_b_quality_dataset.py

echo "== Stage B final evaluation (Frozen Config) =="
python scripts/13_stage_b_final_frozen.py

echo "== Done =="
