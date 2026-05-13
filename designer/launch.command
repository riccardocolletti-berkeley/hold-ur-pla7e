#!/bin/bash
# Double-click in Finder to (re)start the designer and open it in Chrome.
# Resolves the repo root relative to this file so it works regardless of the
# user's current working directory at click time.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "Designer: uv not found on PATH." >&2
    echo "Install uv (https://docs.astral.sh/uv/) and double-click again." >&2
    exit 1
fi

# uv sync is a no-op when the lockfile is already realised on disk.
uv sync >/dev/null

exec uv run python -m designer.run
