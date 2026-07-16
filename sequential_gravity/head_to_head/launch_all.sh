#!/bin/bash
# ============================================================================
# Launches all 4 method jobs (LC/LU/GC/GU) as fully detached background
# processes (nohup setsid ... & disown), matching this repo's existing
# run_d20_*.sh convention. Each job is a single long-running Julia process
# doing its own SEQUENTIAL 9-solve loop (no intra-job parallelism -- see
# HEAD_TO_HEAD_PROMPT.md rationale: KNITRO's floating license is only
# validated to 4-way concurrency, exactly matching these 4 concurrent jobs).
#
# PREREQUISITE: sequential_gravity/head_to_head/shared_starts.jld2 must
# already exist (run generate_shared_starts.jl first) -- this script checks
# and refuses to launch otherwise.
#
# Each job writes its own log to logs/, and a $(method)_ALLDONE sentinel file
# in this directory when its full 9-solve loop finishes. If a job is
# killed/crashes, simply re-running THIS script is safe and resumes every
# job from its last completed solve (each per-solve JLD2 has its own
# skip-if-done check) -- it will NOT re-launch a job whose ALLDONE sentinel
# already exists.
#
#   bash sequential_gravity/head_to_head/launch_all.sh
# ============================================================================
set -euo pipefail

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"

H2H_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$H2H_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$H2H_DIR/shared_starts.jld2" ]; then
    echo "ERROR: $H2H_DIR/shared_starts.jld2 does not exist. Run generate_shared_starts.jl first:"
    echo "  FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true julia -t 19 --project=. sequential_gravity/head_to_head/generate_shared_starts.jl"
    exit 1
fi

mkdir -p "$H2H_DIR/logs"

launch_one() {
    local method="$1" script="$2"
    local sentinel="$H2H_DIR/${method}_ALLDONE"
    local log="$H2H_DIR/logs/${method}_$(date +%Y%m%d_%H%M%S).log"
    if [ -f "$sentinel" ]; then
        echo "[$method] ALLDONE sentinel already present ($sentinel) -- NOT relaunching. Delete it if you want to force a fresh run."
        return
    fi
    echo "[$method] launching detached: $script -> $log"
    FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
      nohup setsid julia -t 19 --project=. "$script" > "$log" 2>&1 &
    disown
    echo "[$method] PID $!"
}

launch_one "lc" "sequential_gravity/head_to_head/run_lc.jl"
launch_one "lu" "sequential_gravity/head_to_head/run_lu.jl"
launch_one "gc" "sequential_gravity/head_to_head/run_gc.jl"
launch_one "gu" "sequential_gravity/head_to_head/run_gu.jl"

echo ""
echo "All non-already-done jobs launched. Check progress with:"
echo "  tail -f $H2H_DIR/logs/*.log"
echo "  ls $H2H_DIR/*_ALLDONE 2>/dev/null   # which jobs have fully finished"
echo "  ls $H2H_DIR/out_{lc,lu,gc,gu}/*.jld2 | wc -l   # completed solves so far (should reach 36)"
