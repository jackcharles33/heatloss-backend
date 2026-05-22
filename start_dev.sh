#!/bin/bash
# ── Local dev startup ──────────────────────────────────────────────────────
# Starts the FastAPI backend on :8000 and opens a note about the frontend.
# The Vite dev server (port 4000) proxies /api/* to localhost:8000, so just
# run `npm run dev` in heatloss-final-1 and everything connects automatically.

set -e
cd "$(dirname "$0")"

# Activate the conda env if not already active
if [[ "$CONDA_DEFAULT_ENV" != "heatloss" ]]; then
  echo "Activating heatloss conda env..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate heatloss
fi

if [ ! -f production_model.joblib ]; then
  echo "ERROR: production_model.joblib not found — run python train_deploy.py first"
  exit 1
fi

echo ""
echo "  Backend  → http://localhost:8000"
echo "  Frontend → cd heatloss-final-1 && npm run dev  (http://localhost:4000)"
echo "  API docs → http://localhost:8000/docs"
echo ""
echo "Starting uvicorn..."
uvicorn predict:app --reload --port 8000 --host 0.0.0.0
