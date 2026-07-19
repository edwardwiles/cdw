# Continuation 10, Section 9, Part C: canonical benchmark (before the delta frontier)

Branch `c10-finalize-architecture`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4-c10-finalize-architecture`.
Real D=20 France-focal data, W=80,000, `demand.mit.edu`, `JULIA_NUM_THREADS=20`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, Julia 1.12.6, KNITRO 14.2.0. Run
through `full_aod_diag/d4_exact/c10_d20_production_driver.jl` (post Part A
wiring: structured Hessian-callback moment materialization + BLAS-gemv
KKT/moment-residual swap, both re-verified in
`docs/fullA_D20_final_architecture_report.md`). Starting point: the calibration
point perturbed to `gp0*1.01` (same starting point workstream `c10-prod-wiring`'s
own smoke test used), upper branch, delta=1, `draw_seed=20260719`.

**Purpose**: per the brief, this is a sanity check that the fully-assembled
architecture actually works end-to-end BEFORE committing real wall-clock time
to the delta=0.1/1/2/5 production frontier (Section 10) -- not a performance
claim in its own right.

## 1. Exact (hard-max) cold value evaluation

```
[1. exact cold value eval] wall=16.184s  Delta_dual=0.2308841490034606  inner_status=0  screen_status=screen_passed
```

Includes the screen (pairwise+witness, both pass silently at this near-feasible
starting point) and a full cold KNITRO inner-dual solve with `obj.x` reset to
NaN (no warm start). `ctx` build itself took 54.6s in this run (first-process
JIT/compile cost dominates; `screen_setup_wall = (pairwise=0.125s, witness=2.32s)`
matches the previously-reported ~2.2-2.5s total screen setup, paid once per
context).

## 2. Warm inner solve at the same point

```
[2. warm inner solve, same pt] wall=0.588s  Delta_dual=0.2308841490034606  inner_status=0
```

Same Delta_dual as the cold solve (0.2308841490034606, identical to 16 digits
shown), confirming warm-starting from the just-computed dual state reproduces
the same converged point far faster (0.588s vs 16.184s, ~27.5x) -- exactly the
expected warm-start behavior this driver relies on for every subsequent
in-KNITRO-loop evaluation.

## 3. Full outer composite gradient evaluation

```
[3. full outer gradient] wall=9.412s  norm(gfull)=97.68020700609671  tie_fallback=false
```

This is the FIRST gradient call in a fresh process (cold FD-bandwidth cache,
JIT still warming up for the gradient-specific code paths), hence noticeably
slower than the ~2.9-3.07s "warm-cache steady-state" figure workstream
`c10-prod-wiring`'s BLAS audit report measured for a REPEATED gradient call in
an already-warm process -- consistent, not a regression (that report's own
number was explicitly the steady-state in-KNITRO-loop case, not a first call).

## 4. Real ~20-50 outer-iteration segment: SR1 vs L-BFGS side-by-side

Two `run_profile_checkpointed` runs, SAME starting point, SAME 480s wall-clock
budget, differing only in `hessopt_tag` (`"sr1"` -> `csw_outer_wallclock_sr1.opt`,
`hessopt=3`; `"lbfgs"` -> `csw_outer_wallclock_lbfgs.opt`, `hessopt=6`):

```
[SR1]   wall_ext=460.8s  n_eval=111  n_grad_calls=27  knitro_status=-101  knitro_iter=26
[SR1]   best: Delta=0.19291663303205261 found_at_eval=111 t_elapsed=457.4s
[LBFGS] wall_ext=486.2s  n_eval=112  n_grad_calls=35  knitro_status=-401  knitro_iter=34
[LBFGS] best: Delta=0.18518680056506007 found_at_eval=110 t_elapsed=470.1s
```

SR1 landed 26 outer iterations (within the requested ~20-50 range) over 460.8s;
L-BFGS landed 34 over 486.2s and reached a lower (better) Delta_dual. Both
terminated on the wall-clock budget (`knitro_status` codes -101/-401, both
outer-level KNITRO "time limit reached" variants, not a solver failure --
`n_eval`/`knitro_iter` kept climbing throughout, consistent with hitting
`maxtime_real` rather than converging to a stationary point). Peak RSS: 5.51GB
(SR1) / 6.42GB (L-BFGS) -- far below any memory-incident concern at this W;
grew roughly linearly with outer iterations as expected (each iterate's
checkpoint/trace bookkeeping is O(1) per iterate, the dominant growth is
ordinary KNITRO/Julia GC churn, not a leak).

See `docs/fullA_D20_final_architecture_report.md`'s SR1-vs-L-BFGS section for
the interpretation (kept SR1 as the default pending a closer multi-seed look,
per this task's explicit "don't over-invest" instruction).

## 5. Exact cold-recheck of the SR1 segment's terminal point

Run `c10_canonical_coldrecheck.jl`, a GENUINELY SEPARATE Julia process (its own
`julia` invocation): loaded the SR1 run's final (`stage_complete`) checkpoint
(`g=0.9976395165999713`, `knitro_iter=26`, `n_eval=111`), re-seeded
`Random.seed!(draw_seed)`, rebuilt `ctx` completely from scratch, and
re-evaluated `evaluate_fullA_screened` at the checkpoint's own `(g, zfree)`
point with **`warm=false`** (`obj.x` reset to NaN -- a genuinely cold inner
solve, NOT reusing the checkpoint's saved dual warm-start state; a stricter
check than the checkpoint/resume mechanism's own "RESUME VALIDATION", which
deliberately reinjects the saved dual state and so only tests checkpoint
serialization fidelity, not path-independence of the converged point):

```
[COLD RECHECK] wall=16.943s inner_status=0 screen_status=screen_passed
[COLD RECHECK] recomputed: Delta_dual=0.192916633032049      gravity=7.458768845614028e-18  kkt=3.159431116728228e-12  |moment_resid|=1.926861225882341e+00
[COLD RECHECK] checkpoint:  Delta_dual=0.192916633032053      gravity=7.458768845614028e-18  kkt=3.589474317777786e-12  |moment_resid|=1.926861225882341e+00
[COLD RECHECK] |diff|:      Delta_dual=3.525e-15  gravity=0.000e+00  kkt=4.300e-13  moment_resid_norm=0.000e+00
```

**Delta_dual reproduces to 3.5e-15 (machine precision), gravity_value and
moment_resid_norm bit-identical, KKT residual differs by 4.3e-13** (both
values themselves sit at the ~1e-12 solver-tolerance noise floor, as discussed
in `docs/fullA_D20_final_architecture_report.md`). This confirms the
26-outer-iteration SR1 segment's terminal point is a genuine, reproducible
feasible point -- not an artifact of one particular warm-start path through
KNITRO -- when re-derived completely from scratch in a fresh process.

## Verdict

The fully-assembled architecture (Part B of the finalize-architecture report)
works end-to-end at real D=20/W=80,000 scale: a cold exact value evaluation,
a warm re-evaluation, a full outer gradient, a real multi-iteration outer
optimization segment, and an independent cold re-derivation of that segment's
terminal point all completed successfully and agree to floating-point
precision. Peak memory (6.42GB at 34 outer iterations) gives no indication of
the W=800,000 memory-blowup pattern seen earlier in this investigation --
though per that incident's own lesson, W=800,000 specifically should still get
its own small-probe memory check before any real run at that scale, which this
benchmark (run entirely at W=80,000) does not itself provide.

**Driver for Section 10's real delta=0.1/1/2/5 frontier runs**:
`full_aod_diag/d4_exact/c10_d20_production_driver.jl`, functions
`run_profile_checkpointed(...)` (fixed-g, A-block-only profile stage) and
`run_polish_checkpointed(...)` (joint constrained `(gamma',A)` polish stage) --
same names, same file, unchanged from workstream `c10-prod-wiring`'s own
naming, now carrying this task's two Part A wins by default (no extra flag
needed -- the swap is inside `compressed_live.jl`/`infeasibility_screen.jl`/
`oracle_fast.jl`, which the driver already depended on).

## Files

New (all under `full_aod_diag/d4_exact/`): `c10_canonical_benchmark.jl` (steps
1-4 above), `c10_canonical_coldrecheck.jl` (step 5).

Checkpoint artifacts: `results/fullA_d4/c10_finalize_canonical/sr1/`,
`results/fullA_d4/c10_finalize_canonical/lbfgs/`,
`results/fullA_d4/c10_finalize_canonical/sr1_coldrecheck/`.

Raw logs (scratchpad, not committed): benchmark stdout captured during this
task's run.
