#!/bin/bash
# Generates provenance.txt and SHA256SUMS_2026-07-24.txt for the fixed-Frechet
# post-omit-ROW port-prep deliverable package. Run from the branch worktree root.
set -euo pipefail
cd "$(dirname "$0")/.."

{
  echo "=== Fixed-Frechet post-omit-ROW port-prep: provenance ==="
  echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "branch: $(git branch --show-current)"
  echo "HEAD: $(git rev-parse HEAD)"
  echo "base (production/fullA-exact): $(git rev-parse production/fullA-exact)"
  echo ""
  echo "=== commits on this branch (base..HEAD) ==="
  git log --oneline production/fullA-exact..HEAD
  echo ""
  echo "=== diff --stat vs base ==="
  git diff --stat production/fullA-exact..HEAD
  echo ""
  echo "=== working tree status ==="
  git status --short
  echo ""
  echo "=== environment ==="
  julia --version 2>/dev/null || echo "julia: not on PATH in this shell"
  echo "hostname: $(hostname)"
  echo "load average at manifest time: $(uptime)"
} > docs/provenance_2026-07-24.txt

find docs/*.md full_aod_diag/d4_exact/cm_frechet_*.jl full_aod_diag/d4_exact/frechet_reference_targets.jl \
     full_aod_diag/d4_exact/run_frechet_upper.jl full_aod_diag/d4_exact/test_frechet_*.jl \
     full_aod_diag/d4_exact/diag_frechet_*.jl full_aod_diag/d4_exact/test_flexible_cm_regression_smoke.jl \
     docs/test_logs docs/provenance_2026-07-24.txt -type f 2>/dev/null | sort | xargs sha256sum > docs/SHA256SUMS_2026-07-24.txt

echo "Wrote docs/provenance_2026-07-24.txt and docs/SHA256SUMS_2026-07-24.txt"
wc -l docs/SHA256SUMS_2026-07-24.txt
