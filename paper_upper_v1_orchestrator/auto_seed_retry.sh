#!/bin/bash
# paper_upper_v1: autonomous overnight seed-generation retry loop.
#
# If a full-scale generate_seeds.jl run doesn't find [seeds].number_of_starts accepted seeds
# (n_accepted < requested, stop_reason=:attempt_limit -- generate_multistart_seeds NEVER
# hand-selects substitutes, per its own docstring), this halves A_scale/gp_scale and retries,
# up to MAX_RETRIES times, before finally launching run_paper_upper_bounds.jl for real (which
# will itself hard-refuse to proceed to Phase I unless the accepted count is complete -- see the
# "Hard-gate Phase I launch" commit).
#
# Never uses rm/rf: generate_multistart_seeds's own write_seed_set OVERWRITES seeds/manifest.jls
# and seeds/seeds/S<k>/... IN PLACE on every call (plain "w"-mode opens / Serialization.serialize,
# both truncate-on-write) -- no cleanup step is ever needed between retries, by construction.
#
# Determinism note: attempt_rng_seed depends on (master_rng_seed, attempt_id, manifest_digest),
# and manifest_digest does NOT include A_scale/gp_scale -- so attempt #k draws the EXACT SAME
# underlying unit-direction vector across every scale tried here, just multiplied by a smaller
# radius each retry. Smaller scale (closer to calibration) should never make acceptance WORSE.
#
# Usage:
#   ./auto_seed_retry.sh protocols/paper_upper_v1.toml /bbkinghome/edav/repo_scratch/paper_upper_v1

set -uo pipefail   # deliberately NOT -e: every step's exit status is inspected explicitly

TOML_PATH="$1"
CAMPAIGN_ROOT="$2"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_RETRIES=6
RETRY_MAX_ATTEMPTS=40   # cap per-scale attempts on retries (the frozen manifest's own
                        # max_attempts=200 stays in effect for the FIRST/full-scale try only --
                        # see below) so a bad scale fails fast rather than burning hours per level.

export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=10
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH="/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}"

cd "$SRC_DIR" || exit 1
mkdir -p "$CAMPAIGN_ROOT"

log() { echo "[auto_seed_retry $(date -u +%FT%TZ)] $*"; }

py_get() {
    # NOTE: the default `python3` on this system is 3.9 (no tomllib, stdlib since 3.11) --
    # /usr/bin/python3.11 confirmed present and used here explicitly. Confirmed live 2026-08-08
    # that silently falling back to plain `python3` here breaks this ENTIRE script on its very
    # first call, which would have been discovered only after the user was already asleep.
    /usr/bin/python3.11 -c "import tomllib; print(tomllib.load(open('$TOML_PATH','rb'))$1)"
}

halve_scales_and_cap_attempts() {
    python3 - "$TOML_PATH" "$RETRY_MAX_ATTEMPTS" <<'PYEOF'
import sys, re
path, cap = sys.argv[1], sys.argv[2]
with open(path) as f:
    text = f.read()
def halve(m):
    val = float(m.group(2))
    return f'{m.group(1)}{val/2:.6g}'
text = re.sub(r'(\nA_scale\s*=\s*)([0-9.]+)', halve, text, count=1)
text = re.sub(r'(\ngp_scale\s*=\s*)([0-9.]+)', halve, text, count=1)
text = re.sub(r'(\nmax_attempts\s*=\s*)([0-9]+)', lambda m: f'{m.group(1)}{cap}', text, count=1)
with open(path, "w") as f:
    f.write(text)
PYEOF
}

reset_and_freeze() {
    python3 -c "
import re
path='$TOML_PATH'
text=open(path).read()
text=re.sub(r'protocol_sha = \"[^\"]*\"', 'protocol_sha = \"PENDING_COMMIT\"', text, count=1)
open(path,'w').write(text)
"
    git add "$TOML_PATH"
    git commit -m "Auto-retry (unattended overnight): revise seed perturbation scale" >/dev/null
    bash "$SRC_DIR/paper_upper_v1_orchestrator/freeze_protocol_source.sh" "$TOML_PATH"
}

TARGET=$(py_get "['seeds']['number_of_starts']")
log "target=$TARGET accepted starts required"

for attempt in $(seq 1 "$MAX_RETRIES"); do
    A=$(py_get "['seeds']['A_scale']")
    G=$(py_get "['seeds']['gp_scale']")
    MA=$(py_get "['seeds']['max_attempts']")
    log "=== seed attempt $attempt: A_scale=$A gp_scale=$G max_attempts=$MA ==="
    SEED_LOG="$CAMPAIGN_ROOT/seed_attempt_${attempt}.log"
    julia --project="$SRC_DIR" "$SRC_DIR/paper_upper_v1_orchestrator/generate_seeds.jl" "$TOML_PATH" "$CAMPAIGN_ROOT" > "$SEED_LOG" 2>&1
    N_ACCEPTED=$(grep -oE 'n_accepted=[0-9]+' "$SEED_LOG" | tail -1 | grep -oE '[0-9]+')
    log "  result: n_accepted=${N_ACCEPTED:-unknown} (log: $SEED_LOG)"
    if [ -n "${N_ACCEPTED:-}" ] && [ "$N_ACCEPTED" -ge "$TARGET" ]; then
        log "=== SUCCESS: $N_ACCEPTED/$TARGET accepted. Launching run_paper_upper_bounds.jl (Phase I). ==="
        exec julia --project="$SRC_DIR" "$SRC_DIR/run_paper_upper_bounds.jl" \
            --protocol "$TOML_PATH" --campaign-root "$CAMPAIGN_ROOT"
    fi
    if [ "$attempt" -eq "$MAX_RETRIES" ]; then
        break
    fi
    log "  shortfall (${N_ACCEPTED:-unknown}/$TARGET) -- halving A_scale/gp_scale, capping max_attempts=$RETRY_MAX_ATTEMPTS, retrying"
    halve_scales_and_cap_attempts
    reset_and_freeze
done

log "=== FAILED: exhausted $MAX_RETRIES retries without reaching $TARGET accepted seeds."
log "NOT launching Phase I. Manual investigation needed -- see $CAMPAIGN_ROOT/seed_attempt_*.log"
exit 1
