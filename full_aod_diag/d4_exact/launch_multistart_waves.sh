#!/bin/bash
# ================================================================================================
# PQ-ONLY upper-bound multistart campaign, reusing the seeds the reproducible five-family campaign
# (protocols/paper_upper_v1.toml) already generated. That TOML is FROZEN: read, never written.
#
# WAVE STRUCTURE. The five-family campaign ran 2 starts x 5 families = 10 simultaneous Phase-I jobs
# over 5 waves. This runs TWO families (PQ standalone and CM+PQ) across the FIRST 5 seeds, which is
# again 10 simultaneous jobs and the SAME 100-core footprint -- one wave, not five:
#
#            campaign (5 families)        this (PQ + CMPQ)
#   per wave 2 starts x 5 fam = 10 jobs   5 starts x 2 fam = 10 jobs
#   cores    10 jobs x 10 = 100           10 jobs x 10 = 100      <- identical footprint
#   waves    5                            1 (over the first 5 seeds)
#
# PQ_FAMILIES / PQ_SEED_COUNT / PQ_STARTS_PER_WAVE control this. Defaults: both families, first 5
# seeds, 5 starts per wave -> a single wave of 10 jobs. Setting PQ_SEED_COUNT=10 gives 2 waves and
# covers all 10 seeds.
#
# [concurrency] is otherwise honoured exactly: julia_threads_per_process=10, cores_per_process=10,
# openblas_num_threads=1, omp_num_threads=1, and cpu_affinity_policy="disjoint_taskset_per_process"
# (each process pinned to its own disjoint 10-core range, so the 5 concurrent chains never contend).
#
# RUNTIME. 4 delta cells x 180 min = 12 h per chain; chains within a wave run concurrently, so
# ~12 h per wave and ~24 h total. Stages skip on an existing checkpoint, so re-running resumes.
#
#   usage: screen -dmS pq_multistart bash launch_pq_multistart_waves.sh <L> [campaign_root]
#          PQ_DRY_RUN=1 bash launch_pq_multistart_waves.sh <L>     # preflight only, runs nothing
# ================================================================================================
set -uo pipefail

PQ_L="${1:?usage: launch_pq_multistart_waves.sh <L> [campaign_root]}"
WORKTREE=/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10
SEED_ROOT=/bbkinghome/edav/repo_scratch/paper_upper_v1/seeds/seeds
CAMPAIGN_ROOT="${2:-/bbkinghome/edav/repo_scratch/pq_multistart_upper_$(date -u +%Y%m%d)/L${PQ_L}}"
DELTAS="${PQ_DELTAS:-0.1,0.5,1.0,2.0}"
STARTS_PER_WAVE="${PQ_STARTS_PER_WAVE:-5}"
FAMILIES="${PQ_FAMILIES:-PQ CMPQ}"               # space-separated; each seed runs once per family
SEED_COUNT="${PQ_SEED_COUNT:-5}"                 # use only the first N seeds (S0..S{N-1})
CORES_PER_PROC="${PQ_CORES_PER_PROC:-10}"        # [concurrency].cores_per_process
FIRST_CORE="${PQ_FIRST_CORE:-0}"                 # base of the disjoint taskset ranges
DRY_RUN="${PQ_DRY_RUN:-0}"

mkdir -p "$CAMPAIGN_ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$CAMPAIGN_ROOT/waves_${STAMP}.log"
log() { echo "[waves $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

log "=== PQ-only multistart campaign (screen=${STY:-<none>}) ==="
log "L=$PQ_L  deltas=$DELTAS  starts/wave=$STARTS_PER_WAVE  cores/proc=$CORES_PER_PROC"
log "seed root:     $SEED_ROOT"
log "campaign root: $CAMPAIGN_ROOT"

# ---- preflight: seeds must exist, and the tree must be clean so the frozen source is real -------
ALL_SEEDS=($(ls -d "$SEED_ROOT"/S* 2>/dev/null | sort -V))
if [ "${#ALL_SEEDS[@]}" -eq 0 ]; then log "ERROR: no seeds under $SEED_ROOT"; exit 2; fi
SEEDS=("${ALL_SEEDS[@]:0:$SEED_COUNT}")
log "found ${#ALL_SEEDS[@]} seeds, using the first ${#SEEDS[@]}: $(basename -a "${SEEDS[@]}" | tr '\n' ' ')"
log "families: $FAMILIES"
FAMARR=($FAMILIES)
for f in "${FAMARR[@]}"; do
    case "$f" in PQ|CMPQ) ;; *) log "ERROR: unknown family '$f' (expected PQ or CMPQ)"; exit 2 ;; esac
done
# CMPQ needs L to divide CM's grid size (50): the shared-mass argument requires the PQ cutoffs to
# lie on CM's grid. The chain runner hard-errors too; catching it here avoids burning a wave.
if [[ " $FAMILIES " == *" CMPQ "* ]] && [ $(( 50 % PQ_L )) -ne 0 ]; then
    log "ERROR: CMPQ requires L | 50 (valid: 2, 5, 10, 25); L=$PQ_L does not divide 50."
    exit 2
fi
for s in "${SEEDS[@]}"; do
    [ -f "$s/economic_seed.jls" ] || { log "ERROR: $s has no economic_seed.jls"; exit 2; }
done

# A dirty tree is fatal for a real launch (git archive only sees committed content, so the frozen
# snapshot would not be the code under test) but must NOT block the preflight -- another agent may
# legitimately be mid-edit in this shared worktree, and you still want to be able to check the plan.
DIRTY=$(git -C "$WORKTREE" status --porcelain | wc -l)
if [ "$DIRTY" -ne 0 ]; then
    if [ "$DRY_RUN" != "0" ]; then
        log "WARNING: worktree has $DIRTY uncommitted change(s). Fine for preflight; a REAL launch"
        log "         will refuse until they are committed."
        git -C "$WORKTREE" status --porcelain | sed 's/^/           /' | tee -a "$LOG"
    else
        log "ERROR: worktree has $DIRTY uncommitted change(s). Commit first -- git archive only sees"
        log "       committed content, so the frozen snapshot would not be the code you tested."
        git -C "$WORKTREE" status --porcelain | sed 's/^/         /' | tee -a "$LOG"
        exit 2
    fi
fi
COMMIT=$(git -C "$WORKTREE" rev-parse HEAD)
SRC_DIR="$CAMPAIGN_ROOT/_source"
if [ "$DRY_RUN" != "0" ]; then
    log "preflight: would freeze source @ $COMMIT into $SRC_DIR"
elif [ ! -d "$SRC_DIR" ]; then
    log "freezing source @ $COMMIT"
    mkdir -p "$SRC_DIR"
    git -C "$WORKTREE" archive "$COMMIT" | tar -x -C "$SRC_DIR" || exit 3
    echo "$COMMIT" > "$CAMPAIGN_ROOT/frozen_commit.txt"
else
    log "reusing frozen source, commit $(cat "$CAMPAIGN_ROOT/frozen_commit.txt" 2>/dev/null)"
fi

NWAVES=$(( (${#SEEDS[@]} + STARTS_PER_WAVE - 1) / STARTS_PER_WAVE ))
JOBS_PER_WAVE=$(( STARTS_PER_WAVE * ${#FAMARR[@]} ))
log "plan: ${#SEEDS[@]} starts x ${#FAMARR[@]} families / $STARTS_PER_WAVE starts per wave = $NWAVES wave(s)"
log "      $JOBS_PER_WAVE jobs per wave x $CORES_PER_PROC cores = $(( JOBS_PER_WAVE * CORES_PER_PROC )) cores; box has $(nproc)"
for ((w=0; w<NWAVES; w++)); do
    lo=$(( w * STARTS_PER_WAVE )); hi=$(( lo + STARTS_PER_WAVE - 1 ))
    [ $hi -ge ${#SEEDS[@]} ] && hi=$(( ${#SEEDS[@]} - 1 ))
    names=""; for ((i=lo; i<=hi; i++)); do names="$names $(basename "${SEEDS[$i]}")"; done
    log "      wave $((w+1)):$names  x [$FAMILIES]"
done

if [ "$DRY_RUN" != "0" ]; then
    log "PQ_DRY_RUN=$DRY_RUN -- preflight only, nothing launched. Re-run without it to start."
    exit 0
fi

# ---- environment: protocol [concurrency] --------------------------------------------------------
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1      # CLAUDE.md hard cap + [concurrency].openblas_num_threads
export OMP_NUM_THREADS=1           # [concurrency].omp_num_threads
export JULIA_NUM_THREADS="$CORES_PER_PROC"
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH="/opt/shared_sw/knitro/13.0.1/lib:${LD_LIBRARY_PATH:-}"
log "env: JULIA_NUM_THREADS=$JULIA_NUM_THREADS OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1"

cd "$SRC_DIR" || exit 4

for ((w=0; w<NWAVES; w++)); do
    lo=$(( w * STARTS_PER_WAVE )); hi=$(( lo + STARTS_PER_WAVE - 1 ))
    [ $hi -ge ${#SEEDS[@]} ] && hi=$(( ${#SEEDS[@]} - 1 ))
    log "================= WAVE $((w+1))/$NWAVES : seeds $lo..$hi ================="
    pids=(); slot=0
    for ((i=lo; i<=hi; i++)); do
        sdir="${SEEDS[$i]}"; sid=$(basename "$sdir")
        for fam in "${FAMARR[@]}"; do
            odir="$CAMPAIGN_ROOT/$fam/$sid"; mkdir -p "$odir"
            c0=$(( FIRST_CORE + slot * CORES_PER_PROC )); c1=$(( c0 + CORES_PER_PROC - 1 ))
            log "  launching $fam/$sid on cores $c0-$c1 -> $odir/chain.log"
            taskset -c "$c0-$c1" julia --project=. \
                full_aod_diag/d4_exact/run_multistart_seed_chain.jl \
                "$fam" "$sdir" "$odir" "$PQ_L" "$DELTAS" > "$odir/chain.log" 2>&1 &
            pids+=($!); slot=$(( slot + 1 ))
        done
    done
    log "  wave $((w+1)) launched: ${#pids[@]} chains, pids ${pids[*]}"
    fail=0
    for p in "${pids[@]}"; do wait "$p" || { log "  chain pid $p exited nonzero"; fail=$((fail+1)); }; done
    log "  wave $((w+1)) complete ($fail chain(s) exited nonzero)"
done

# ---- aggregate ----------------------------------------------------------------------------------
SUM="$CAMPAIGN_ROOT/summary_L${PQ_L}.txt"
{
    echo "Multistart upper-bound campaign, families=[$FAMILIES], L=$PQ_L, deltas=$DELTAS"
    echo "frozen commit: $(cat "$CAMPAIGN_ROOT/frozen_commit.txt" 2>/dev/null)"
    echo "seeds from:    $SEED_ROOT  (generated by the five-family paper_upper_v1 campaign)"
    echo
    echo "--- seed qualification (seeds were qualified for the OTHER five families, not these) ---"
    for fam in "${FAMARR[@]}"; do
        for s in "${SEEDS[@]}"; do
            sid=$(basename "$s"); v="$CAMPAIGN_ROOT/$fam/$sid/seed_verdict.txt"
            [ -f "$v" ] && cat "$v" || echo "family=$fam seed_id=$sid verdict=NO_VERDICT_FILE"
        done
    done
    echo
    echo "--- per-chain results ---"
    for fam in "${FAMARR[@]}"; do
        for s in "${SEEDS[@]}"; do
            sid=$(basename "$s"); f="$CAMPAIGN_ROOT/$fam/$sid/chain_summary.txt"
            if [ -f "$f" ]; then sed "s/^/[$fam $sid] /" "$f"; else echo "[$fam $sid] (no chain_summary.txt)"; fi
        done
    done
} > "$SUM"
log "summary: $SUM"
log "=== campaign complete ==="
