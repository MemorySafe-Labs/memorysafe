#!/usr/bin/env bash
# Launch MemorySafe live buffer demo (Streamlit).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating .venv and installing requirements (first run only)..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

echo "→ self-test"
.venv/bin/python demo_engine.py

echo "→ streamlit (http://localhost:8501)"
exec .venv/bin/streamlit run demo_streamlit.py