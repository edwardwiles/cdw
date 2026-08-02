#!/bin/bash
# Generates SHA256_MANIFEST.txt over every deliverable this session produced (docs/CSVs/JSON at
# worktree root + the new full_aod_diag/d4_exact/*_2026-08-01.jl files). Run from the worktree root.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="SHA256_MANIFEST.txt"
{
  git ls-files --others --exclude-standard -- '*_2026-08-01*.md' '*_2026-08-01*.csv' '*_2026-08-01*.json'
  git ls-files -- '*_2026-08-01*.md' '*_2026-08-01*.csv' '*_2026-08-01*.json' 2>/dev/null || true
  find full_aod_diag/d4_exact -maxdepth 1 -name '*_2026-08-01.jl' -newer PROFILED_UNRESTRICTED_OUTER_AB_MASTER_2026-08-01.md 2>/dev/null || true
  find full_aod_diag/d4_exact -maxdepth 1 -name '*_2026-08-01.jl'
} | sort -u | xargs -I{} sha256sum "{}" > "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") files)"
