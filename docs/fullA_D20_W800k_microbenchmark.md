# Full-A D=20 real-data (W=800,000): memory-safety result (partial, Phase 2)

Continuation 9, Phase 2 (W=800,000 half). This document covers only the
**memory-safety and single-point-feasibility** portion of Phase 2's W=800,000
scope. The full 4-point breakdown / thread-sweep the standing brief also asks
for at W=800,000 is deliberately **deferred** until after Phase 3 lands its
compressed-mode and `build_lfix_base_cache` speedups (in progress as of this
writing) — benchmarking the current dense-only path in full detail at
W=800,000 right before it changes would be a poor use of the ~70-190s/call
cost this scale carries. This document will be extended (or superseded by a
combined post-Phase-3 W=80k+W=800k comparison) once that work lands.

## Why this ran at all: the incident

The very first attempt to characterize W=800,000 memory usage (pre-fix,
`needs_outer_moment_jacobian=true` default inherited from `context_scaled.jl`'s
diagnostic convention) was killed after climbing to **~780GB VmHWM and still
rising** — headed past 1TB on a shared 3TB machine — caught via a user-flagged
server-wide memory alert (23.5%+ and growing). Root cause and fix (commit
`bd313a2`) are documented in `docs/fullA_D20_W80k_microbenchmark.md` §0: `jac_h`
is a dense `W × (nTotalMoments+2) × l_full` tensor, `≈109GB` at W=80,000 and
`≈1.09TB` at W=800,000 at the bad default. The production convention
(`needs_outer_moment_jacobian=false`, matching `run_fullA_D4/D10_production.jl`)
removes this tensor entirely.

## Post-fix result

Script: `full_aod_diag/d4_exact/c9_w800k_memsafety_probe.jl`. One context build
+ one cold `evaluate_fullA` call at the natural-theta (calibration) point,
monitored externally via `/proc/<pid>/status` `VmHWM` polling with an automatic
200GB safety kill armed (never triggered).

| | value |
|---|---|
| `d20_real_setup(W=800000)` wall (cold, JIT paid) | 187.02s |
| Julia live heap after build+GC | 8.486 GB |
| `evaluate_fullA` cold eval wall | 73.49s |
| inner_status | 0 (clean convergence) |
| Delta_dual | 0.00026237 |
| gravity_raw | -3.38e-16 (machine zero) |
| Julia live heap after eval+GC | 8.483 GB |
| **Peak process VmHWM (external, whole run)** | **16.55 GB** |

**Delta_dual shrinks 10x from W=80,000's 0.00259 to W=800,000's 0.00026** — both
near-zero, consistent with a correctly-specified calibration point (population
divergence ≈0) and consistent with sampling noise shrinking as `W` grows
(matches the standing memory note "W=8000 silently understates kappa for
delta>=1 vs W>=80k" — here the effect runs the other direction at a genuinely
near-zero population value, and the 10x draw increase produces something close
to the theoretically expected `~sqrt(10)`-ish noise reduction in the same
direction, not a contradiction of that note).

**Headline: W=800,000 is now safe and cheap** — 16.55GB peak (0.5% of this
machine's 3TB) is trivial; dozens of independent W=800,000 contexts could run
concurrently with room to spare, memory-wise (wall-clock/CPU contention is a
separate question, not addressed by this single-point probe). This resolves
the core blocker Phase 9 was gating on. The remaining open question is
wall-clock cost at scale (187s cold build + 73s cold eval for ONE point;
Phase 8's pilot and Phase 9's cost projection will need the full profile,
not just this safety check, to answer "how many W=800,000 points are
practical").

## What this document does NOT yet cover (deferred, see above)

- Full 4-outer-point breakdown (calibration / gravity-tangent / upper-branch /
  lower-branch) at W=800,000, matching `docs/fullA_D20_W80k_microbenchmark.md`'s
  §2-4.
- Component-level `@prof` breakdown of the value callback and `L_fix` gradient
  at W=800,000.
- Thread-count sweep (1/5/10/20) at W=800,000.
- Warm-vs-cold comparison at W=800,000 (only cold was measured here).

## Reproduce

```bash
cd /bbkinghome/edav/gravity_robustness/gravity-fullA-d4
source .knitro_env.sh
export JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
julia --project=. full_aod_diag/d4_exact/c9_w800k_memsafety_probe.jl
```

Raw log: `results/fullA_d4/bd313a2/c9_w800k_memsafety/probe_log.txt`.
