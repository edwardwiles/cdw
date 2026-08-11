#!/bin/bash
# ================================================================================================
# Launcher for the pairwise-quantile-independence production run (version B: fixed Frechet-z
# cutoffs + free bin masses), 2026-08-10.
#
# Run it INSIDE screen, never bare:
#   screen -dmS pq_prod_L5 bash <this> 5 frechet_theoretical
#
# What it does:
#   1. FREEZES the source. `git archive` of the exact commit into <campaign_root>/_source, so the
#      running campaign is pinned to a tree that cannot change under it while someone keeps editing
#      the worktree. (The live paper_upper_v1 campaign uses the same _source discipline.)
#   2. Runs run_pairwise_quantile_production.jl over the protocol's delta grid, per-stage
#      checkpointed.
#   3. Retries on a nonzero exit, bounded. Retrying is SAFE and resumes rather than restarts: every
#      stage skips itself if its checkpoint exists, so a retry picks up where the crash left off.
#
# KNITRO: pinned to 13.0.1, which is what every gate in this session was validated under. The live
# paper_upper_v1 campaign runs 14.2.0; do not mix them into one comparison without re-gating.
# ================================================================================================
set -uo pipefail

PQ_L="${1:?usage: launch_pairwise_quantile_production.sh <L> <cutoff_source> [deltas_csv]}"
CUTOFF_SOURCE="${2:?usage: launch_pairwise_quantile_production.sh <L> <cutoff_source> [deltas_csv]}"
DELTAS="${3:-0.1,0.5,1.0,2.0}"

WORKTREE=/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10
CAMPAIGN_ROOT=/bbkinghome/edav/repo_scratch/pq_freemass_production_2026-08-10/L${PQ_L}_${CUTOFF_SOURCE}
SRC_DIR="$CAMPAIGN_ROOT/_source"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$CAMPAIGN_ROOT/run_${STAMP}.log"
MAX_ATTEMPTS=6

mkdir -p "$CAMPAIGN_ROOT"
log() { echo "[launch $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

log "=== pairwise-quantile production launch (screen=${STY:-<none>}) ==="
log "L=$PQ_L  cutoff_source=$CUTOFF_SOURCE  deltas=$DELTAS"
log "campaign root: $CAMPAIGN_ROOT"

# ---- 1. freeze the source at the current commit --------------------------------------------
if [ ! -d "$SRC_DIR" ]; then
    COMMIT=$(git -C "$WORKTREE" rev-parse HEAD)
    DIRTY=$(git -C "$WORKTREE" status --porcelain | wc -l)
    log "freezing source from $WORKTREE @ $COMMIT (uncommitted files: $DIRTY)"
    if [ "$DIRTY" -ne 0 ]; then
        # git archive only sees committed content, so a dirty tree would silently run OLD code.
        log "ERROR: worktree has $DIRTY uncommitted change(s). Commit first -- otherwise the frozen"
        log "       snapshot would not be the code you just tested."
        exit 2
    fi
    mkdir -p "$SRC_DIR"
    git -C "$WORKTREE" archive "$COMMIT" | tar -x -C "$SRC_DIR" || exit 3
    echo "$COMMIT" > "$CAMPAIGN_ROOT/frozen_commit.txt"
    log "source frozen: $(find "$SRC_DIR" -name '*.jl' | wc -l) .jl files"
else
    log "reusing existing frozen source ($SRC_DIR), commit $(cat "$CAMPAIGN_ROOT/frozen_commit.txt" 2>/dev/null)"
fi

# ---- 2. environment -------------------------------------------------------------------------
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1      # CLAUDE.md: hard cap, always, under Julia
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=16
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH="/opt/shared_sw/knitro/13.0.1/lib:${LD_LIBRARY_PATH:-}"
log "env: JULIA_NUM_THREADS=$JULIA_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS KNITRO=$KNITRODIR"

cd "$SRC_DIR" || exit 4

# ---- 3. run, with bounded resume-retries -----------------------------------------------------
attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    log "--- attempt $attempt/$MAX_ATTEMPTS ---"
    julia --project=. full_aod_diag/d4_exact/run_pairwise_quantile_production.jl \
        "$PQ_L" "$CUTOFF_SOURCE" "$CAMPAIGN_ROOT" "$DELTAS" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        log "run completed cleanly (attempt $attempt)"
        break
    fi
    log "run exited rc=$rc on attempt $attempt; checkpoints are on disk, retrying (resumes, not restarts)"
    attempt=$((attempt + 1))
    sleep 30
done

log "=== launcher finished (last rc=${rc:-na}) ==="
log "summary file: $CAMPAIGN_ROOT/summary_L${PQ_L}.txt"
