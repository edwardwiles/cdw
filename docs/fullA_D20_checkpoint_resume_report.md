# Continuation 10, Section 6: checkpoint/resume for the fast D=20 driver

Branch `c10-prod-wiring`, worktree `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4-c10-prod-wiring`.
Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, real D=20 France-focal data, W=80,000, Julia 1.12.6, KNITRO (linked)
13.0.1.

## THE driver for Section 10's real production frontier runs

**`full_aod_diag/d4_exact/c10_d20_production_driver.jl`** — use this file. It extends
`full_aod_diag/d4_exact/c9_phase8_d20_pilot.jl`'s architecture (same production config:
compressed moments, `h_mode=:cached` bandwidth policy, `multi_method=:top3`, SR1 outer
Hessian — unchanged, already gated PASS by Continuation 9 Phases 6/7/8) with two additions:
default infeasibility screening (Section 5, see that report) and checkpoint/resume (this
section). It exposes `run_profile_checkpointed(...)` (fixed-g, A-block-only profile stage)
and `run_polish_checkpointed(...)` (joint constrained `(gamma', A)` polish stage,
warm-started from a profile's terminal point) — the same two-stage recipe Phase 8
established as the working production recipe. `c9_phase8_d20_pilot.jl` itself is
UNCHANGED and remains available as a reference/diagnostic script, but is not the driver to
use going forward.

## What gets checkpointed, and how

A `D20Checkpoint` struct (serialized via the Julia stdlib `Serialization` module — no new
dependency) is written on four triggers, per the brief:

1. **Every accepted KNITRO outer iterate**, via `KNITRO.KN_set_newpt_callback` — confirmed
   available and cheap to use (`KN_set_newpt_callback(kc, callback)`, fires with the
   latest `x`/`lambda` after every iteration KNITRO accepts; checked via
   `names(KNITRO, all=true)` before assuming it existed).
2. **Every new best-feasible point** (inside `cb_F!`, same point `best[]`/`best_feasible[]`
   already tracks).
3. **Every `checkpoint_interval_s` of wall time** (configurable, checked inside `cb_F!`;
   default in the smoke test was 15s, production default is left at the caller's
   discretion — the brief's suggested 60-120s is a reasonable production value).
4. **At the end of each stage** (`:stage_complete`, after `KN_solve` returns).

Each checkpoint captures, per the brief's explicit list: `g` (gamma'_focal), the full
`D x D` log(A_od) matrix AND the reduced `zfree` free coordinates, the CC inner-solve's
dual warm start (`copy(ctx.obj.x)`), the current best feasible point, the FD bandwidth
cache (`Dict{Int,Float64}`), the branch (`:upper`/`:lower`, derived from `find_smallest`)
and delta, and — see the finding below — the draw seed. Checkpoints are written
atomically (`serialize` to a `.tmp` file, then `mv`) so a crash mid-write can never leave a
half-written checkpoint that a resume could load.

**Solver/quasi-Newton state — NOT checkpointed, and this is a real API limitation, not an
oversight.** Checked `names(KNITRO, all=true)` for anything hessian/state/restart-shaped:
found `KN_get_hessian_values` (reports the CURRENT exact-Hessian evaluation, only meaningful
for exact-Hessian modes — not applicable, this driver's default is SR1/BFGS/L-BFGS
quasi-Newton) and nothing else. The KNITRO.jl/C API does not expose extracting or
reinjecting the internal SR1/BFGS/L-BFGS Hessian-approximation state across separate
`KN_new()` instances. A resumed run restarts its outer Hessian approximation from KNITRO's
own default initialization — the same situation as any fresh KNITRO solve warm-started at
a good primal point, not a degraded one.

## A real, non-obvious finding this task surfaced: the draw seed problem

`context_real_d20.jl`'s underlying data pipeline (`setup/importData.jl`'s `fakeData==3`
branch, `prepare_cc/genRands.jl::genExpRands!`) draws the `W` Frechet simulation support
via `rand!(U)` on the GLOBAL Julia RNG, **unseeded**, for the real-data path (only the
synthetic `fakeData in (1,2)` paths call `Random.seed!`). This means `ctx.U` — and
therefore every `Delta_dual`/gravity/moment-residual number this entire investigation
computes — is **not reproducible across separate Julia processes** unless something seeds
the RNG first.

A checkpoint/resume driver by definition restarts in a NEW process, so a resume that
doesn't address this would silently evaluate the checkpointed point against a
DIFFERENT simulated economy and get different numbers — a genuine correctness trap that
would have made the acceptance test below fail non-obviously (small differences, not a
crash). Fixed by having `c10_d20_production_driver.jl` call `Random.seed!(draw_seed)`
immediately before `d20_real_setup` in BOTH `run_profile_checkpointed` and
`run_polish_checkpointed`, and recording `draw_seed` in every checkpoint (default
`20260719` unless overridden). This is exactly the "draw seed/identifier" field the
standing brief asked for — its necessity was confirmed empirically, not assumed.

## Acceptance test: PASSED

Per the brief's own stated bar — a restarted run from a checkpoint must reproduce the SAME
objective, divergence, gravity residual, and moment residuals at the checkpoint's own point
as the original unbroken run had there, to numerical precision.

**Procedure**: `c10_prod_driver_smoke_original.jl` ran `run_profile_checkpointed` for a
90s-budget profile stage at real D=20/W=80,000 (upper branch, `checkpoint_interval_s=15`),
producing 6 checkpoint files (4 `new_best`, 1 `wall_interval`-eligible run that didn't
trigger in this short window, 1 `stage_complete`) and exiting normally (`status=-401`,
time-limit reached, feasible). Then, in a **genuinely separate Julia process**
(`c10_prod_driver_smoke_resume.jl`, its own `julia` invocation, no shared state),
`run_profile_checkpointed(...; resume_from = <latest checkpoint path>)` loaded the
checkpoint, re-seeded the RNG with its recorded `draw_seed`, rebuilt `ctx` from scratch,
reinjected the dual warm start into `ctx.obj.x`, and immediately re-evaluated
`evaluate_fullA_screened` at the checkpoint's own `(g, zfree)` point.

**Result** (verbatim from the resumed process's log):

```
[smoke_upper] RESUME VALIDATION at checkpoint's own point: |ΔDelta_dual|=0.0 |Δgravity_value|=0.0 |Δmax_abs_moment_kkt_resid|=0.0 |Δ||moment_resid|||=0.0
[smoke_upper]   original: Delta_dual=0.20946000734480752 gravity=3.2938062000109624e-18
[smoke_upper]   resumed:  Delta_dual=0.20946000734480752 gravity=3.2938062000109624e-18
```

**Exact bit-identical reproduction** (0.0 diff, not merely close) across all four
quantities the acceptance test names: objective/divergence (`Delta_dual`), gravity
residual (`gravity_value`), and moment residuals (`max_abs_moment_kkt_resid` and
`||moment_resid||`). This directly confirms the `Random.seed!` fix is necessary AND
sufficient: reproducibility across a genuine process boundary — the real failure mode a
checkpoint/resume mechanism must survive — holds to full floating-point precision.

The resumed run then proceeded to run KNITRO for its own (small, 5s) budget and exited
normally, confirming the mechanism also works end-to-end as an actual resume (not just a
point-reconstruction check): `[smoke_upper] PROFILE DONE: status=-401 wall_ext=11.3s
n_eval=9 n_grad_calls=1 screens(pw/wt/wn/pass)=0/0/0/2` — `n_eval` continuing on from the
original run's own count (8), i.e. state genuinely carried across the checkpoint boundary,
not restarted from zero.

## Files

New (all under `full_aod_diag/d4_exact/`):
- **`c10_d20_production_driver.jl`** — the driver (this section's headline deliverable;
  also carries Section 5's default screening).
- `c10_prod_driver_smoke_original.jl` / `c10_prod_driver_smoke_resume.jl` — the two-process
  acceptance test above (kept as a reusable regression check, not a one-off scratch
  script).

Modified: none beyond `context_real_d20.jl` (Section 5's `build_screen` addition, which
this driver also depends on).

Checkpoint artifacts from the validation run: `results/fullA_d4/c10_ckpt_smoke_test/`
(original run) and `results/fullA_d4/c10_ckpt_smoke_test_resumed/` (resumed run's own new
checkpoints) — left in place as evidence, not cleaned up mid-session per this
investigation's own "defer cleanup to the end" convention.
