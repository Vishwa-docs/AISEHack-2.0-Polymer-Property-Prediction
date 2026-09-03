#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11}"
export JUPYTER_DATA_DIR="$run_dir/.jupyter_data"
mkdir -p "$JUPYTER_DATA_DIR"

"$python_bin" - <<'PY'
import sys
assert sys.version_info[:3] == (3, 11, 7), sys.version
print("Using Python", sys.version.split()[0])
PY

if [[ ! -x "$run_dir/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$run_dir/.venv"
fi
"$run_dir/.venv/bin/python" -m pip install --disable-pip-version-check -r "$run_dir/requirements-isolated.txt"
"$run_dir/.venv/bin/python" -m ipykernel install --prefix "$run_dir/.venv" --name ppp311-isolated --display-name "PPP isolated (Python 3.11.7)"
"$run_dir/.venv/bin/python" "$run_dir/verify_environment.py"
