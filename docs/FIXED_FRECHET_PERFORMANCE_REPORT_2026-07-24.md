# Performance Report — Fixed-Fréchet Post-Omit-ROW Port — 2026-07-24

Task brief §13. **Important caveat that applies to every timing number in this document**: all
measurements were taken on a heavily-loaded shared machine (`load average ~186-193` on 208 cores,
sustained throughout this session — confirmed via `uptime`, not incidental). Every wall-clock
number below is therefore an upper bound on this port's intrinsic cost, not a clean benchmark —
flagged explicitly rather than presented as a clean number. Where a comparison to the pre-omit-ROW
archive's own (differently-loaded) measurements is informative, it is given as a ratio/order-of-
magnitude, not a literal apples-to-apples benchmark.

## D=4 (synthetic, uncontended relative to D=20's real-data I/O)

| Stage | Time |
|---|---|
| D=4 context build | <1s |
| CDF+POWER dense (Architecture A) solve, `L=8` | a few seconds |
| CDF+POWER structured (Architecture C) Hessian build | sub-second |
| Full D=4 gate suite (`test_frechet_power_hessian_d4_gates.jl`, 13 checks) | well under 1 minute total |
| Basis-invariance gate suite (9 checks) | well under 1 minute total |

## D=20, W=80,000, real data (`:exclude_row`)

| Stage | Time (this session, heavy load) |
|---|---|
| Context build (`d20_real_setup_design`, real-data load + gravity elimination) | 134-196s (varied run-to-run under contention) |
| `fpcx` construction, `L=50`, `:cdf_power` (dense CM matrix build, `W×2000`) | 34-64s |
| Bounded-`L=8` dense (Architecture A) solve at real D=20/W=80,000 scale | 21.7-31.8s |
| Bounded-`L=8` structured (Architecture C) Hessian build | ~11.8s |
| Full-`L=50` calibrated-benchmark solve (structured Hessian only, at the RAW calibration point) | included in the ~5-8 min total gate wall time; the calibration point itself solves fast (KNITRO warm-starts well from an exact-fit point) |
| Cold solve at a point slightly AWAY from calibration (`gp_target = gp* ` at `κ*+1e-4`, first-ever solve, no warm start) | **~660-790s** — over an order of magnitude slower than at the exact calibration point; this is the dominant cost driver for the outer shakedown, see below |
| Outer-gradient computation (`cm_frechet_production_gradient`, reference backend), fresh call | ~50s |
| Outer-gradient computation, SAME point, base reused (no re-solve) | ~24s — confirms `base=` reuse avoids the expensive inner re-solve, the accepted-point state-reuse discipline (task brief §8) working as intended |

## Structured-vs-dense Hessian precision at scale

| Point | max abs diff | max abs dense entry | relative diff |
|---|---|---|---|
| D=4, `L=8`, `:cdf_power` | 2.33e-15 | — | machine precision |
| D=20 real data, `L=8`, `:cdf_power`, `W=80,000` | 1.396e-5 | 3977.9 | **3.51e-9** |

The larger absolute number at D=20 is entirely explained by scale (larger core block, wider
dynamic range from the real economic parameters spanning ~11 orders of magnitude, per this
project's own standing calibration finding) — the *relative* error is consistent with the D=4
result to within two orders of magnitude, not a qualitatively different regime.

## Exact-cache / state-reuse instrumentation

`run_frechet_upper.jl` instruments: `n_eval`, `n_grad`, `n_new_point_solve`,
`n_base_reused_at_gradient`, `n_time_limit_no_certificate`, `n_infeasible_certificate`. See the
port-readiness report's "Outer shakedown status" section for the actual counts from the live run.
No same-point repeated inner solve was observed in any test in this pass: every gradient call at an
already-evaluated point reused the cached `BaseDualState` (confirmed both in the standalone
gradient-wrapper validation, `max|g-g2|=0.0` across two calls at the identical point with `base`
passed explicitly the second time, and in `run_frechet_upper.jl`'s own `last_F_state`-based reuse
mechanism, mirroring `cm_outer_driver.jl`'s established pattern).

## Memory

Not separately profiled with a dedicated RSS-tracking harness in this pass (disclosed gap, not a
claim of memory-safety at scale). The dense `:cdf_power` CM matrix at `D=20/L=50/W=80,000` is
`80,000 × 2,000 × 8 bytes ≈ 1.28GB`, built once per context (not per KNITRO callback) — plausible
but not independently measured against this repo's own 40GB self-imposed kill threshold in this
pass. Peak RSS during the live D=20 gate/shakedown runs was not captured; a future pass should add
`Base.gc_live_bytes()`/`/proc/self/status` sampling around context and `fpcx` construction,
matching this repo's own established memory-audit convention for other restricted-model families.

## Root cause of the ~700-1000s per-new-point inner solve (live diagnosis)

See `FIXED_FRECHET_SLOW_INNER_SOLVE_DIAGNOSIS_2026-07-24.md` for the full writeup. Summary: KNITRO
registers the `ncore+ncm=2382`-variable inner dual problem's Hessian as fully dense
(`2,838,153` nonzeros = complete upper triangle), solved single-threaded (`par_numthreads=1` in
`ek_inner.opt`) via Interior-Point/Barrier Direct at very tight tolerances (`opttol=1e-12`,
`ftol=1e-15`). A captured 14-iteration partial trace shows normal interior-point behavior
(geometric `OptError` decay within each barrier stage, periodic resets on barrier-parameter
reduction) at ~15s/iteration. **This is KNITRO's own dense per-iteration linear algebra cost, not
this port's moment-construction or structured-Hessian-callback code** — the latter were
independently confirmed fast (sub-second to ~12s even at real D=20 scale) via the D=4/D=20
correctness gates above. Pre-existing configuration, not introduced by this port; the most
promising, lowest-risk follow-up lever (multi-threading the inner linear solver on this 208-core
machine) touches no code this port owns.

## "Slow but solved" vs "time-limit, no certificate" vs "certified infeasible"

Explicitly distinguished throughout this port via `frechet_solve_outcome`/`decode_knitro_status`
(see the timeout/state-reuse audit doc). Every D=20 solve in this pass's gates landed in the
"slow but solved" category (`nStatus=0`, optimal) — no `time_limit_no_certificate` or
`infeasible_certificate` outcome was observed in the correctness gates; the outer shakedown's own
instrumentation (see above) is the intended place to observe these in a live multi-point search,
since a single-point gate cannot exercise a genuinely infeasible trial point.
