# Continuation 10, Section 9: finalize the production architecture

Branch `c10-finalize-architecture` (forked from `diag/fullA-d4-exact` @ `97cdd80`,
the tip after this continuation's three parallel workstreams -- `c10-prod-wiring`,
`c10-chunked-hessian`, `c10-qmc-is` -- were merged), worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4-c10-finalize-architecture`.
Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, real D=20 France-focal data, W=80,000, Julia 1.12.6, KNITRO
14.2.0.

This task had three parts: (A) wire in the two validated-but-deferred wins from
this continuation's own workstreams, (B) confirm/assemble the final production
architecture, (C) run a canonical before-the-frontier benchmark. Part C is
reported separately in `docs/fullA_D20_canonical_benchmark.md`; this report
covers (A) and (B).

## Part A.1 -- structured moment construction wired into the ACTUAL Hessian callback

**Call-graph trace** (not guessed): `c10_d20_production_driver.jl`'s
`run_profile_checkpointed`/`run_polish_checkpointed` both route every evaluation
through `screened_eval` -> `evaluate_fullA_screened` (`infeasibility_screen.jl`)
-> (for `moment_representation=:compressed`, the driver's actual config) ->
`evaluate_fullA_screened_compressed` (`infeasibility_screen.jl`) ->
`inner_loop_KNITRO_compressed` (`compressed_live.jl`), which registers
`_callbackEvalH_inner_compressed!` as KNITRO's inner-dual Hessian callback. THAT
function (`compressed_live.jl` line ~163) is where the dense `W x (D^2+1)` G
matrix is lazily materialized, once per inner solve, from the already-built
`CompressedFactual`. This is the actual target the task's brief pointed at --
the SAME `_callbackEvalH_inner_compressed!` is used regardless of whether the
call arrived via `evaluate_fullA_fast_compressed` (`compressed_live.jl`, reached
directly, not via the screen) or `evaluate_fullA_screened_compressed`
(`infeasibility_screen.jl`, the driver's actual hot path) -- both share this one
Hessian-callback registration.

**What changed**: `structured_moment_build.jl` gained one new function,
`materialize_dense_factual_structured!(Gview, cf; use_ger=false)`, an exact
drop-in replacement for `compressed_moments.jl::materialize_dense_factual!`
with an IDENTICAL signature -- it internally calls the already-validated
`structured_coeffs`/`structured_fill_chunk!` (workstream `c10-chunked-hessian`,
unmodified) over the full `1:cf.W` draw range. `use_ger=false` (broadcast)
matches that workstream's own finding that broadcast beats `BLAS.ger!` by ~30%
for this D^2-wide rank-one fill.

Three call sites were swapped from `materialize_dense_factual!` to
`materialize_dense_factual_structured!`:
- `compressed_live.jl`'s `_callbackEvalH_inner_compressed!` (the actual
  Hessian callback -- the primary target).
- `compressed_live.jl`'s `evaluate_fullA_fast_compressed` tail (a lazy-
  materialize fallback for the rare case KNITRO converges before ever calling
  the Hessian callback).
- `infeasibility_screen.jl`'s `evaluate_fullA_screened_compressed` tail (the
  SAME fallback, independently duplicated there per this codebase's own
  "provably non-interfering duplicate tail" convention).

All three do the IDENTICAL formula; leaving some on the old builder while
others used the new one would have reintroduced exactly the kind of drift this
codebase's own duplicated-tail discipline is designed to prevent, so all three
were swapped together, not just the one the brief named.

`structured_moment_build.jl` is now `include`d by `c10_d20_production_driver.jl`
(right after `compressed_moments.jl`) and by `test_compressed_live_integration.jl`
/ `test_infeasibility_screen.jl` (so those existing regression scripts keep
working); dozens of older one-off Continuation 8/9 benchmark scripts that also
`include` `compressed_live.jl`/`infeasibility_screen.jl` were left untouched
(out of scope -- frozen diagnostic artifacts of already-completed phases, not
part of the production call graph going forward).

**Re-verification, end-to-end through the ACTUAL driver** (not just workstream
1's own isolated D=4 benchmark script): ran the SAME real D=20/W=80,000 point
(`gp0*1.01`, calibration-point `zfree0`, `draw_seed=20260719`) through
`evaluate_fullA_screened(...; moment_representation=:compressed, ...)` twice --
once against the pre-change code at `diag/fullA-d4-exact` tip `97cdd80`
(`c10_equivalence_probe_OLD.jl`, run in the base worktree), once against this
branch's post-change code (`c10_finalize_equivalence_probe.jl`):

| quantity | OLD (pre-change) | NEW (this branch) | agreement |
|---|---|---|---|
| Delta_dual (cold) | 0.230884149003460 | 0.230884149003461 | 15 digits |
| gravity_value | 6.361827297601819e-18 | 6.361827297601819e-18 | bit-identical |
| max_abs_moment_kkt_resid | 2.904e-12 | 1.734e-12 | both at the ~1e-12 noise floor (see note) |
| norm(moment_resid) | 0.7729772165483534 | 0.7729772165482959 | 13 digits |
| zeta* | -0.271820564654310 | -0.271820564654421 | 12 digits |
| norm(lambda*) | 6.438001438663997 | 6.438001438676348 | 10 digits |
| outer gradient norm | 97.68020700591120 | 97.68020700609671 | 10 digits |
| winner_hash | 3279919630480572855 | 3279919630480572855 | identical |

Agreement is at floating-point-reordering precision throughout, not bit-for-bit
-- expected and consistent with workstream 1's own D=4 finding (differences
"at machine-precision level", 1.78e-15), scaled up slightly at real D=20/W=80,000
where more summation reordering occurs. The KKT residual itself is a quantity
that is supposed to be ~0 at a converged optimum, so both runs' O(1e-12) values
are consistent with "converged to the solver's own tolerance", not a
discrepancy between old and new code.

Additionally, a DIRECT isolated check at the real D=20/W=80,000 `CompressedFactual`
this driver actually builds (not a D=4 synthetic one): `max|G_new - G_old| =
1.137e-13` -- machine precision, confirming the swapped-in function reproduces
`materialize_dense_factual!`'s exact output at production scale.

## Part A.2 -- BLAS-gemv KKT/moment-residual swap wired in

**Where**: the identical hand-rolled nested-loop pattern (workstream
`c10-prod-wiring`'s BLAS audit, `docs/fullA_D20_blas_audit_report.md`) appeared
in THREE places, not the two the report names -- `oracle_fast.jl::evaluate_fullA_fast`
(the dense tail), `infeasibility_screen.jl::evaluate_fullA_screened_compressed`
(the driver's actual compressed-screened tail), AND `compressed_live.jl::evaluate_fullA_fast_compressed`
(the same duplicated-tail pattern noted above). All three were swapped, for the
same drift-avoidance reason as Part A.1.

Two small helper functions, `kkt_residual_blas`/`moment_resid_blas`, were added
to `oracle_fast.jl` (included before the other two files in every include chain
in this codebase, and forward-referenced at call time regardless): both use
`transpose(view(G,...)) * v` via `LinearAlgebra.mul!`, which dispatches to BLAS
gemv for dense `Float64` arrays with no copy -- the exact candidate the audit
benchmarked.

**Re-verification**: a synthetic W=80,000 x d=402 formula check (matching the
audit's own construction) confirmed `kkt_residual_blas`/`moment_resid_blas`
against the original nested loops: `|diff| = 3.6e-17` (KKT) and `1.5e-16`
(moment residual) -- both at the reported 1e-16/1e-17 floating-point
summation-order noise level, not a discrepancy.

**Screen false-positive re-confirmation** (Part A's other explicit requirement):
re-ran workstream `c10-prod-wiring`'s own `c10_screen_wiring_validate.jl`,
adapted in two small ways (`c10_finalize_screen_wiring_revalidate.jl`): switched
`moment_representation` from `:dense` to `:compressed` throughout (the ORIGINAL
script only ever exercised `oracle_fast.jl`'s dense tail, never
`infeasibility_screen.jl::evaluate_fullA_screened_compressed` / `compressed_live.jl`'s
Hessian callback -- i.e. never the code this task modified), and added the new
`structured_moment_build.jl` include. Result, reproducing the SAME rejection
counts as the original report even though every evaluation now runs through the
newly-wired structured+BLAS code path:

| stage | rejections (of 40 adversarial trials) |
|---|---|
| pairwise certificate | 3 |
| witness | 1 |
| winner-scan (full) | 0 |
| passed through to real solve | 36 |

**0/4 false positives** (every rejected point cross-checked against the
ground-truth full winner-scan). Class 1 (10 feasible calibration/small-perturbation
points): 10/10 passed, 0 rejected. Timing: feasible point 4.02s (real solve
runs) vs. rejected point 0.000064s (62,867x cheaper) -- consistent with the
original report's own >50,000x finding.

## Part B -- the final architecture

**The production architecture for the D=20/W=80,000 delta-frontier runs is**:
exact hard-max values throughout (`mode==:hard` is the only mode
`evaluate_fullA_fast`/`evaluate_fullA_screened`/`evaluate_fullA_fast_compressed`
implement -- both dense and compressed paths error loudly on any other `mode`,
confirmed by reading `oracle_fast.jl` line ~188 and `compressed_live.jl` line
~285; there is no smoothed/homotopy code anywhere in this driver's call graph);
full-A gravity elimination (`gravity_elimination.jl`, unchanged, included by the
driver); the compressed winner-form objective/gradient representation
(`compressed_moments.jl`/`compressed_cc_inner.jl`, O(W·D) vs dense O(W·D^2));
the dense exact inner Hessian, now built via the NEW structured (rank-one +
winner-scatter) construction from Part A.1 above (not chunked -- workstream
`c10-chunked-hessian` found chunking a wash-to-14%-slower and it stays
rejected); the fast composite `L_fix` outer gradient
(`composite_gradient_fast.jl::composite_gradient_at_fast`, threaded, with FD
bandwidth caching -- see below); exact infeasibility screening
(pairwise-certificate -> witness -> destination winner-scan, built once per
context and threaded through every evaluation via `screened_eval`, workstream
`c10-prod-wiring` Section 5, re-confirmed above); checkpoint/resume
(`D20Checkpoint`, workstream `c10-prod-wiring` Section 6, unchanged by this
task); cached FD bandwidths with staleness-aware periodic re-validation
(`bandwidth_cache_policy.jl::BandwidthCachePolicy`, unchanged); and
coordinate-level threading over 20 Julia threads (`Threads.@threads :static`
inside `composite_gradient_at_fast`/`composite_gradient_at_fast_buffered`,
already the standing default).

**Two things changed by THIS task, both re-verified above**: the dense
Hessian-callback materialization now uses the structured (rank-one +
winner-scatter) construction instead of the old generic nested-loop dense
builder (Part A.1), and the KKT-residual/moment-residual post-solve reductions
now use BLAS gemv instead of hand-rolled nested loops (Part A.2). Everything
else in the list above was already true before this task and is confirmed
still working after these two changes (equivalence + screen re-validation
above).

**L_fix buffer-reuse** (workstream `c10-prod-wiring`'s other validated-but-not-
adopted win, `lfix_buffer_reuse.jl`, ~1.06x wall / ~1.23x fewer allocations,
fixing a real `Threads.threadid()` > `Threads.nthreads()` bug): left
NOT wired into `c10_d20_production_driver.jl`'s default `cb_G!` in this task.
That win is orthogonal to Part A's two deferred wins (a different file,
`composite_gradient_at_fast` vs `composite_gradient_at_fast_buffered`, not
mentioned in this task's explicit Part A scope) -- flagged here as still
available (`composite_gradient_at_fast` -> `composite_gradient_at_fast_buffered`
in both `cb_G!`s) for a future pass, consistent with workstream
`c10-prod-wiring`'s own report describing it as "a one-line swap ... once the
coordinating session wants the extra ~6%."

## SR1 vs L-BFGS: side-by-side verdict

One real profile-step run each (`csw_outer_wallclock_sr1.opt` /
`csw_outer_wallclock_lbfgs.opt`, `hessopt=3` vs `hessopt=6`), SAME starting
point (calibration `zfree0`, `gp0*1.01`, upper branch, delta=1), SAME 480s
wall-clock budget, real D=20/W=80,000 data (full numbers in
`docs/fullA_D20_canonical_benchmark.md`):

| | wall_ext | n_eval | n_grad_calls | outer iters | best Delta_dual | time-to-best |
|---|---|---|---|---|---|---|
| SR1 (current default) | 460.8s | 111 | 27 | 26 | 0.19291663303205261 | 457.4s |
| L-BFGS | 486.2s | 112 | 35 | 34 | 0.18518680056506007 | 470.1s |

**In this single run, L-BFGS reached MORE outer iterations (34 vs 26) and a
LOWER (better) Delta_dual in comparable wall time** (~5% more wall time for 31%
more outer iterations and a materially better objective) -- a preliminary
signal against the standing SR1 default, not a settled verdict. Per this
task's explicit "one real profile-step run is enough, don't over-invest"
instruction, this is a single, non-multi-seed comparison at one starting point
and one budget -- it does not have the statistical weight of Continuation 9's
own Phase 6/7/8 gating that established SR1 as the default. **Recommendation:
keep SR1 as `c10_d20_production_driver.jl`'s default (`hessopt_tag` defaults
to `"sr1"`, unchanged by this task) for the upcoming delta-frontier runs, but
flag this result for a closer look** (e.g. a multi-seed or multi-starting-point
comparison) before the NEXT round of production decisions if L-BFGS's edge
holds up. Both `csw_outer_wallclock_sr1.opt` and `csw_outer_wallclock_lbfgs.opt`
already exist and the driver already accepts `hessopt_tag` as a kwarg, so
switching is a one-line change to any future call, not a wiring task.

## Files

Modified (all under `full_aod_diag/d4_exact/`): `structured_moment_build.jl`
(new `materialize_dense_factual_structured!`), `oracle_fast.jl` (new
`kkt_residual_blas`/`moment_resid_blas`, both call sites swapped),
`compressed_live.jl` (2 materialize call sites + 2 KKT/moment-residual sites
swapped), `infeasibility_screen.jl` (1 materialize call site + 2 KKT/moment-
residual sites swapped), `c10_d20_production_driver.jl` (new include),
`test_compressed_live_integration.jl` / `test_infeasibility_screen.jl` (new
include, so these existing regression scripts keep working after the swap).

New (all under `full_aod_diag/d4_exact/`): `c10_finalize_equivalence_probe.jl`
(this task's end-to-end old-vs-new equivalence probe, run against this branch),
`c10_equivalence_probe_OLD.jl` (the same probe's pre-change counterpart, run
in the base `gravity-fullA-d4` worktree against tip `97cdd80` for direct
comparison -- NOT committed to this branch, lives only in the base worktree as
a throwaway comparison artifact), `c10_finalize_screen_wiring_revalidate.jl`
(re-validation of the infeasibility screen through the newly-wired compressed
path), `c10_canonical_benchmark.jl` / `c10_canonical_coldrecheck.jl` (Part C,
see `docs/fullA_D20_canonical_benchmark.md`).

Raw logs: `results/fullA_d4/c10_finalize_canonical/` (SR1/LBFGS checkpoints +
cold-recheck checkpoint from the canonical benchmark run).
