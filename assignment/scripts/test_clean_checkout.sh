#!/usr/bin/env bash
# test_clean_checkout.sh
#
# Prove that the repository runs from git-tracked files ONLY.
#
# Creates a detached git worktree (containing exactly the tracked files of
# the given ref), provisions a fresh environment from the tracked
# pyproject.toml + uv.lock with `uv sync`, then runs the full pytest suite
# inside it -- including the end-to-end pipeline test on the committed fake
# dataset. Any dependence on untracked or deleted files (code, data, or an
# unpinned dependency) fails loudly.
#
# Usage:
#   bash scripts/test_clean_checkout.sh [REF]    # default: HEAD
#
# Requirements:
#   uv (https://docs.astral.sh/uv/) on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REF="${1:-HEAD}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed. See https://docs.astral.sh/uv/" >&2
    exit 1
fi

WORKTREE_DIR="$(mktemp -d /tmp/sa-clean-checkout.XXXXXX)"

cleanup() {
    cd "$PROJECT_ROOT"
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    rm -rf "$WORKTREE_DIR"
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
echo "[INFO] Creating clean worktree of '$REF' at $WORKTREE_DIR"
git worktree add --detach "$WORKTREE_DIR" "$REF"

cd "$WORKTREE_DIR"
echo "[INFO] Provisioning environment from tracked uv.lock..."
# A self-contained venv inside the worktree, built only from tracked files.
unset VIRTUAL_ENV
uv sync --frozen

echo "[INFO] Running pytest in the clean worktree..."
uv run python -m pytest tests -x -q

echo "[OK] Test suite passed using git-tracked files only."
