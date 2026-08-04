#!/bin/bash
# launch_full_campaign_2026-08-04.sh -- top-level launcher for the post-verifier-fix W=100k
# rerun + fresh K=3 campaign (task sections 6-11). Ten disjoint 10-core taskset slots (0-99),
# 10 Julia threads/process, BLAS=8 (this repo's own validated ten-by-ten resource layout).
#
# Topology:
#   6 non-ZC chains (unrestricted/flexible_cm/common_frechet x upper/lower), full delta grid
#   0.01/0.1/0.5/1/2, 1hr budget/delta (35min explore + 25min polish), launched immediately.
#   4 K=3 chains (origin_zc/cm_meanzc x upper/lower), Wave 1 (delta 0.01/0.1/0.5, 2hr/delta),
#   launched CONCURRENTLY with the 6 non-ZC chains (10 processes total, matching task section 6).
#   Wave 2 (delta 1/2) does NOT start until ALL 10 processes above have exited AND the Wave-1
#   K3_WAVE1_FINAL_SEED_REGISTRY.csv snapshot is frozen -- task section 10's explicit barrier.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.juliaup/bin:$PATH"

export CAMPAIGN_W=100000
export CAMPAIGN_MEANZC_K=3
export CAMPAIGN_ORIGINZC_K=3

D4E="full_aod_diag/d4_exact"
RUNCHAIN="$D4E/run_campaign_chain_2026-08-04.sh"
POST_ROOT="/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX"

NONZC_DELTAS=(0.01 0.1 0.5 1 2)
NONZC_EXPLORE=2100   # 35 min
NONZC_POLISH=1500    # 25 min
K3_WAVE1_DELTAS=(0.01 0.1 0.5)
K3_WAVE2_DELTAS=(1 2)
K3_EXPLORE=4500       # 75 min
K3_POLISH=2700        # 45 min

echo "=== Resource pre-check $(date -Iseconds) ==="
nproc --all
free -h
uptime
echo "=== Writing 10 campaign manifests (task section 5) ==="
julia --project=. "$D4E/write_campaign_manifests.jl"

echo "=== Launching 6 non-ZC chains + 4 K=3 Wave-1 chains (10 processes, cores 0-99) ==="
pids=(); labels=()
"$RUNCHAIN" unrestricted    upper  0  9 $NONZC_EXPLORE $NONZC_POLISH "${NONZC_DELTAS[@]}" & pids+=($!); labels+=("unrestricted_upper")
"$RUNCHAIN" unrestricted    lower 10 19 $NONZC_EXPLORE $NONZC_POLISH "${NONZC_DELTAS[@]}" & pids+=($!); labels+=("unrestricted_lower")
"$RUNCHAIN" flexible_cm     upper 20 29 $NONZC_EXPLORE $NONZC_POLISH "${NONZC_DELTAS[@]}" & pids+=($!); labels+=("flexible_cm_upper")
"$RUNCHAIN" flexible_cm     lower 30 39 $NONZC_EXPLORE $NONZC_POLISH "${NONZC_DELTAS[@]}" & pids+=($!); labels+=("flexible_cm_lower")
"$RUNCHAIN" common_frechet  upper 40 49 $NONZC_EXPLORE $NONZC_POLISH "${NONZC_DELTAS[@]}" & pids+=($!); labels+=("common_frechet_upper")
"$RUNCHAIN" common_frechet  lower 50 59 $NONZC_EXPLORE $NONZC_POLISH "${NONZC_DELTAS[@]}" & pids+=($!); labels+=("common_frechet_lower")
"$RUNCHAIN" origin_zc       upper 60 69 $K3_EXPLORE $K3_POLISH "${K3_WAVE1_DELTAS[@]}" & pids+=($!); labels+=("origin_zc_upper_wave1")
"$RUNCHAIN" origin_zc       lower 70 79 $K3_EXPLORE $K3_POLISH "${K3_WAVE1_DELTAS[@]}" & pids+=($!); labels+=("origin_zc_lower_wave1")
"$RUNCHAIN" cm_meanzc       upper 80 89 $K3_EXPLORE $K3_POLISH "${K3_WAVE1_DELTAS[@]}" & pids+=($!); labels+=("cm_meanzc_upper_wave1")
"$RUNCHAIN" cm_meanzc       lower 90 99 $K3_EXPLORE $K3_POLISH "${K3_WAVE1_DELTAS[@]}" & pids+=($!); labels+=("cm_meanzc_lower_wave1")

echo "Launched ${#pids[@]} chains: ${labels[*]}"
echo "PIDs: ${pids[*]}"
mkdir -p "$POST_ROOT"
for i in "${!pids[@]}"; do echo "${labels[$i]}: pid=${pids[$i]}"; done > "$POST_ROOT/pid_record_wave1.txt"

fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "Chain ${labels[$i]} exited nonzero"; fail=1; }
done
echo "=== All 10 processes (6 non-ZC full chains + 4 K=3 Wave-1 chains) finished at $(date -Iseconds), fail=$fail ==="

echo "=== Freezing K3_WAVE1_FINAL_SEED_REGISTRY.csv (barrier before Wave 2) ==="
if ! julia --project=. "$D4E/freeze_k3_wave_registry.jl" "$POST_ROOT/K3_WAVE1_FINAL_SEED_REGISTRY.csv" 0.5; then
  echo "Wave-1 snapshot freeze FAILED -- refusing to launch Wave 2. Fix and re-run Wave 2 manually."
  exit 1
fi
touch "$POST_ROOT/K3_WAVE1_COMPLETE.marker"
echo "K3_WAVE1_COMPLETE.marker written."

echo "=== Launching 4 K=3 Wave-2 chains (cores 60-99, freed from Wave 1) ==="
pids2=(); labels2=()
"$RUNCHAIN" origin_zc upper 60 69 $K3_EXPLORE $K3_POLISH "${K3_WAVE2_DELTAS[@]}" & pids2+=($!); labels2+=("origin_zc_upper_wave2")
"$RUNCHAIN" origin_zc lower 70 79 $K3_EXPLORE $K3_POLISH "${K3_WAVE2_DELTAS[@]}" & pids2+=($!); labels2+=("origin_zc_lower_wave2")
"$RUNCHAIN" cm_meanzc upper 80 89 $K3_EXPLORE $K3_POLISH "${K3_WAVE2_DELTAS[@]}" & pids2+=($!); labels2+=("cm_meanzc_upper_wave2")
"$RUNCHAIN" cm_meanzc lower 90 99 $K3_EXPLORE $K3_POLISH "${K3_WAVE2_DELTAS[@]}" & pids2+=($!); labels2+=("cm_meanzc_lower_wave2")

for i in "${!pids2[@]}"; do echo "${labels2[$i]}: pid=${pids2[$i]}"; done > "$POST_ROOT/pid_record_wave2.txt"

for i in "${!pids2[@]}"; do
  wait "${pids2[$i]}" || { echo "Chain ${labels2[$i]} exited nonzero"; fail=1; }
done
echo "=== Wave 2 finished at $(date -Iseconds), fail=$fail ==="

echo "=== Freezing K3_COMPLETE_RESULT_REGISTRY.csv ==="
if julia --project=. "$D4E/freeze_k3_wave_registry.jl" "$POST_ROOT/K3_COMPLETE_RESULT_REGISTRY.csv" 0.01 0.1 0.5 1 2; then
  touch "$POST_ROOT/K3_CAMPAIGN_COMPLETE.marker"
  echo "K3_CAMPAIGN_COMPLETE.marker written."
else
  echo "Final K3 registry freeze FAILED -- inspect POST_ROOT before declaring K3 complete."
  fail=1
fi

echo "=== CAMPAIGN LAUNCHER DONE $(date -Iseconds) fail=$fail ==="
exit $fail
