#!/usr/bin/env bash
# Use isolated metadata only in the managed workspace where .git is protected.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$project_root/.git-local/objects" ]]; then
    exec git --git-dir="$project_root/.git-local" --work-tree="$project_root" "$@"
fi
exec git -C "$project_root" "$@"
