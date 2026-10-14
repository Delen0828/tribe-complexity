#!/usr/bin/env bash
# Serve the frontend and its API together, from any working directory.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin=""
open_browser=true
server_args=()

usage() {
    cat <<'EOF'
Usage: ./start-visualizer.sh [--port PORT] [--host HOST] [--no-open] [--python PATH]

Starts the visualizer frontend and API on one port (default: 127.0.0.1:8000).
Uses an active environment, a local .venv, or the tribe Conda environment.
Checks dependencies and builds annotation exports if they are missing.
Opens the frontend in your browser; use --no-open on a remote/headless machine.
Keep this terminal open. Press Ctrl+C to stop the server.

Examples:
  ./start-visualizer.sh
  ./start-visualizer.sh --port 8001 --no-open
  ./start-visualizer.sh --python /path/to/environment/bin/python
EOF
}

while (($#)); do
    case "$1" in
        --python)
            if (($# < 2)); then
                echo 'Missing path after --python.' >&2
                exit 2
            fi
            python_bin="$2"
            shift 2
            ;;
        --no-open) open_browser=false; shift ;;
        --help|-h) usage; exit 0 ;;
        *) server_args+=("$1"); shift ;;
    esac
done

if [[ -n "$python_bin" ]]; then
    python_bin="$(command -v -- "$python_bin")" || {
        echo 'The --python interpreter was not found.' >&2
        exit 1
    }
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != base && -x "$CONDA_PREFIX/bin/python" ]]; then
    python_bin="$CONDA_PREFIX/bin/python"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
else
    conda_bin="$(command -v conda || true)"
    if [[ -z "$conda_bin" ]]; then
        for candidate in "${CONDA_EXE:-}" "$HOME/miniforge3/bin/conda" \
            "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
            if [[ -x "$candidate" ]]; then
                conda_bin="$candidate"
                break
            fi
        done
    fi
    if [[ -n "$conda_bin" ]]; then
        python_bin="$("$conda_bin" run -n tribe python -c 'import sys; print(sys.executable)')" || {
            echo 'Could not use the tribe Conda environment. Activate your environment or pass --python PATH.' >&2
            exit 1
        }
    else
        python_bin="$(command -v python3 || command -v python || true)"
    fi
fi

if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo 'Python was not found. Create the tribe environment from environment.yml or pass --python PATH.' >&2
    exit 1
fi
if [[ "$python_bin" != /* ]]; then
    python_bin="$(cd -- "$(dirname -- "$python_bin")" && pwd)/$(basename -- "$python_bin")"
fi

cd -- "$repo_root"
echo "Python: $python_bin"
"$python_bin" - <<'PY'
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', str(Path('tribe/cache/matplotlib').resolve()))
try:
    from visualizer.server import Visualizer
    import matplotlib
    matplotlib.use('Agg')
    from nilearn import plotting
    from nilearn.surface import PolyMesh
except ImportError as error:
    sys.exit(f'Visualizer dependency check failed: {error}\n'
             f'Interpreter: {sys.executable}\n'
             'Use the tribe environment (environment.yml), or pass --python PATH.')
print('Visualizer dependencies OK.')
PY

if [[ ! -f label/output/labels.json ]]; then
    echo 'Building missing annotation exports...'
    "$python_bin" label/build.py
fi

if "$open_browser"; then
    server_args+=(--open)
fi
# exec keeps shutdown simple: Ctrl+C reaches the only server process directly.
exec "$python_bin" -u visualizer/server.py "${server_args[@]}"
