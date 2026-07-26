#!/usr/bin/env bash
# Five-family finish task, Phase 1.1 (2026-07-26): live CLI gate for the unrestricted family's
# repoint to run_polish_checkpointed_unified. Exercises every item task §1.1 lists, through the
# ACTUAL unrestricted_stage_runner.jl entry point (not a reimplementation).
set -uo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ROOT=/bbkinghome/edav/gravity_robustness/worktrees/finish-five-family-optimization-stack-2026-07-26
D4E="$ROOT/full_aod_diag/d4_exact"
OUT="$ROOT/results/phase1_1_gate_2026-07-26"
mkdir -p "$OUT"
cd "$D4E"
FAIL=0
run() {
  local name="$1"; shift
  echo "=== RUN: $name ==="
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "  exit=$rc  log=$OUT/$name.log"
  return $rc
}

# 1. Calibration, upper bound (find_smallest=1), transformed-A default, budget=60s
FIND_SMALLEST=1 julia --project="$ROOT" -t 8 unrestricted_stage_runner.jl "$OUT/ckpt_upper" 1.0 60 calibration none \
  > "$OUT/1_upper_transformedA.log" 2>&1
echo "1_upper_transformedA exit=$?"

# 2. Calibration, lower bound (find_smallest=0), transformed-A default, budget=60s
FIND_SMALLEST=0 julia --project="$ROOT" -t 8 unrestricted_stage_runner.jl "$OUT/ckpt_lower" 1.0 60 calibration none \
  > "$OUT/2_lower_transformedA.log" 2>&1
echo "2_lower_transformedA exit=$?"

# 3. Calibration, explicit legacy-z coordinate mode, budget=60s
FIND_SMALLEST=1 A_COORDINATE_MODE=legacy_z julia --project="$ROOT" -t 8 unrestricted_stage_runner.jl "$OUT/ckpt_legacyz" 1.0 60 calibration none \
  > "$OUT/3_legacyz_explicit.log" 2>&1
echo "3_legacyz_explicit exit=$?"

# 4. Resume from run 1's own unified checkpoint (same coordinate mode/schema) budget=30s
CKPT1=$(ls -t "$OUT/ckpt_upper"/*_unified_latest.jls 2>/dev/null | head -1)
if [ -n "$CKPT1" ]; then
  FIND_SMALLEST=1 julia --project="$ROOT" -t 8 unrestricted_stage_runner.jl "$OUT/ckpt_upper" 1.0 30 resume "$CKPT1" \
    > "$OUT/4_resume_unified.log" 2>&1
  echo "4_resume_unified exit=$? ckpt=$CKPT1"
else
  echo "4_resume_unified SKIPPED: no unified checkpoint found from run 1"
fi

# 5. Refusal test: try to `resume` (unified path) a genuine PRE-EXISTING legacy V4 checkpoint.
LEGACY_CKPT="/bbkinghome/edav/gravity_robustness/unrestricted_overnight_bounds_2026-07-24/upper/delta_1.0/stage_latest.jls"
if [ -f "$LEGACY_CKPT" ]; then
  FIND_SMALLEST=1 julia --project="$ROOT" -t 8 unrestricted_stage_runner.jl "$OUT/ckpt_refuse_test" 1.0 30 resume "$LEGACY_CKPT" \
    > "$OUT/5_legacy_refusal.log" 2>&1
  echo "5_legacy_refusal exit=$? (nonzero + MIGRATION ERROR string expected)"
else
  echo "5_legacy_refusal SKIPPED: no legacy checkpoint found at $LEGACY_CKPT"
fi

# 6. Explicit legacy_profile_resume mode on that SAME legacy checkpoint -- should succeed via old driver.
if [ -f "$LEGACY_CKPT" ]; then
  julia --project="$ROOT" -t 8 unrestricted_stage_runner.jl "$OUT/ckpt_legacy_resume" 1.0 30 legacy_profile_resume "$LEGACY_CKPT" \
    > "$OUT/6_legacy_profile_resume.log" 2>&1
  echo "6_legacy_profile_resume exit=$?"
else
  echo "6_legacy_profile_resume SKIPPED"
fi

echo "=== ALL PHASE 1.1 RUNS DONE ==="
