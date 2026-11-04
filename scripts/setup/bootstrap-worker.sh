#!/usr/bin/env bash
# Install each backend in a separate user-owned worker environment.
set -euo pipefail
root=${1:?Pass a worker-local workspace directory}
source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
requirements_dir="$source_dir/requirements"
if [[ ! -d "$requirements_dir" ]]; then
    requirements_dir=$(CDPATH= cd -- "$source_dir/../../requirements" && pwd)
fi
mkdir -p "$root"
export UV_CACHE_DIR="$root/cache" UV_PYTHON_INSTALL_DIR="$root/python"
if [[ ! -x "$root/tools/bin/uv" ]]; then
    python3 -m venv "$root/tools"
    "$root/tools/bin/pip" install 'uv==0.13.6'
fi
uv="$root/tools/bin/uv"
for backend in xla torchtpu; do
    "$uv" venv --python 3.12 --allow-existing "$root/$backend"
    "$uv" pip install --python "$root/$backend/bin/python" \
        --index-strategy unsafe-best-match -r "$requirements_dir/$backend.txt"
    "$uv" pip freeze --python "$root/$backend/bin/python" > "$root/$backend-freeze.txt"
done
