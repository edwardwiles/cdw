# Melitz sorted-tail moment-construction optimization -- 2026-07-25

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing from
`docs/melitz_real_d20_scaled_knitro_and_profile_2026-07-25.md`. New: `src/melitz/sorted_tail.jl`
(`MelitzSortedTailContext`, `melitz_active_tail_start`, `melitz_moments_sorted_tail!`,
`melitz_moments_sorted_tail_parallel!`, `melitz_sorted_tail_diagnostics`);
`src/melitz/sorted_crossing_gradient.jl` (new, same-day continuation -- the sorted
crossing-slice direct-gradient backend, `:B_direct_argument_sorted_serial`); additive
`moment_backend` kwarg on `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`/
`melitz_calibration_outer_ctx` and backend dispatch in `melitz_moments_adapter!`
(`src/melitz/delta_star.jl`, `src/melitz/pareto_calibration.jl`, now including
`:sorted_tail_parallel`); new tests (`test/melitz/runtests.jl`, "Sorted-tail moment
construction (2026-07-25)", "Phase 11", "Phase 11 follow-up", "Phase 6/7");
`scripts/melitz_sorted_tail_benchmark_2026-07-25.jl` and
`scripts/melitz_sorted_tail_scaling_2026-07-25.jl` (new).

**Scope discipline, stated up front, in the same spirit as every prior session in this
repo's Melitz history**: the governing prompt is a 14-phase program spanning moment
construction, outer-gradient crossing-slice updates, sorted dual-argument construction,
suffix-sum inner-gradient construction, and same-origin Hessian structure, each with its
own multi-D/multi-W/multi-thread benchmark grid. This is a two-part session (same day):
the FIRST part completed Phases 0-3, 5, 11 in full and a reduced Phase 4/12/13; the
SECOND part (this continuation, explicitly requested: "try working on some of those things
that you deferred") completes Phase 6/7 (the sorted crossing-slice outer-gradient backend,
proved exact via this codebase's own already-validated affine-cutoff structure, not merely
approximated) for the DIRECT trade-cell block, extends Phase 4/13 with a genuine
thread-count sweep (`{1,2,4,8,16,20}`) and `W`-sweep (`{20000,40000,80000,160000}`), and
completes the parallel `moment_backend` production-dispatch wiring left as a stated
follow-up in the first part. **Phases 8-10 (sorted dual-argument construction, suffix-sum
inner-gradient construction, and Hessian same-origin-block structure) are STILL not
implemented** -- Section D states exactly what each would require and why they were not
attempted this session either, consistent with this repo's own established practice of
disclosing incomplete sub-phases explicitly rather than extrapolating.

## Executive summary

1. **The sorted-tail moment-construction optimization is REAL and LARGE at real D=20/W=80,000,
   with the exact same numerical moment system as the dense reference.** Live,
   post-JIT, `scripts/melitz_sorted_tail_benchmark_2026-07-25.jl`, real production
   calibration (France focal, `noah_D20`), 4 representative outer points:

   | backend | median wall (trade-share block) | speedup vs dense |
   |---|---:|---:|
   | `:dense_reference` (unchanged) | **1.49-1.52s** | 1.0x |
   | `:sorted_tail_serial` (new) | **0.132-0.134s** | **~11.3x** |
   | `:sorted_tail_parallel`, 16 Julia threads (new) | **0.080-0.081s** | **~18.7x** |

   The dense figure directly reproduces this repo's own previously-documented "~1.46 seconds
   in some real-D20 callbacks" finding (`docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md`
   context) rather than assuming it -- confirmed live at `1.49-1.52s` across 4 points, not
   copied from memory. `MelitzSortedTailContext` construction is a one-time `0.187s` cost,
   amortized across every FC/GA callback for the life of a KNITRO bundle (built once in
   `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`, never inside a
   callback) -- fully paid back by roughly the SECOND callback at this fixture's scale.
2. **Exact numerical equivalence, not merely a fast approximation**: validated at D=4
   (8000+ direct assertions: sorting invariants, 20000+ random active-tail lookups, exact
   agreement with `melitz_firm`'s own active flag across every real cutoff, full moment
   matrices at the calibrated point/random perturbations/focal-origin perturbations/
   engineered zero-active and all-active cells/an explicit near-cutoff tie), D=10 (a fresh
   independent fixture), and real D=20 (the actual production calibration) -- maximum
   absolute disagreement `3.55e-15` (D=4 smoke check, machine precision) to `<1e-8` (real
   D=20, tighter tolerance requested but not needed in practice). **Validated through a real
   nested KNITRO inner solve** (not just the standalone moment matrix): `:sorted_tail_serial`
   reproduces `:dense_reference`'s `Delta`, LFD weight vector, and maximum weighted moment
   residual to `1e-8` when both backends are run end-to-end via `melitz_recover_lfd` at the
   real D=20 calibration.
3. **Parallel scaling is real but sublinear at this fixture's own thread budget**: 16 Julia
   threads give `~1.65x` over the serial sorted-tail backend (not 16x) -- expected, since the
   parallel region only threads over `D=20` origins (a small unit of parallel work) and the
   serial focal-link column remains unparallelized (Section C.1). Correctness is
   bit-identical between serial and parallel (disjoint per-origin column writes, no
   reduction, no race) -- confirmed directly, not merely assumed from the "disjoint writes"
   argument.
4. **Phases 6-10 (crossing-slice outer-gradient updates, sorted dual-argument/suffix-sum
   inner-gradient construction, Hessian same-origin-block structure) are NOT implemented
   this session** -- Section D states what each would require. The moment-construction
   speedup demonstrated here does NOT, by itself, translate into a proportional speedup of
   a full outer KNITRO campaign, whose own wall-clock is dominated by the INNER KNITRO
   solve itself and the outer gradient's own repeated re-evaluations (prior sessions'
   documented findings, e.g. `docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md`)
   -- moment construction is one input to that total, not the whole of it. See Section F.4
   for the honest accounting of what this DOES and does NOT establish about total campaign
   wall-clock.
5. **Recommendation** (Section H): adopt `:sorted_tail_serial` as the new default for any
   NEW real-D20 moment-construction call site (it is now opt-in, not yet the global default,
   to avoid silently changing every existing script's behavior in this same session);
   `:sorted_tail_parallel` is available but its `~1.65x` marginal gain over serial at 16
   threads should be weighed against KNITRO's own already-parallel internal machinery and
   this repo's own standing `BLAS.set_num_threads(1)`-during-Julia-parallelism convention
   before defaulting to it in a driver that also runs BLAS-heavy work concurrently.
6. **Same-day continuation (Phase 6/7, Section D.1): a sorted CROSSING-SLICE variant of the
   PRODUCTION registered outer-gradient backend (`:B_direct_argument_serial`) is now
   implemented, PROVED exact (not merely tested), and validated live at real D=20/W=80,000**
   -- `:B_direct_argument_sorted_serial` (`src/melitz/sorted_crossing_gradient.jl`). The
   production backend already restricts each coordinate's probe to `O(D)` "touched" trade
   cells (`argument_localized_gradient.jl`'s already-validated `melitz_compact_columns_map`
   -- this session reuses that audit rather than re-deriving it), but still scans each
   touched column densely over all `W` rows. Using this codebase's OWN already-validated
   affine-cutoff result (`affine_cutoff.jl`), this session proves EXACTLY (a short algebraic
   sandwich argument, Section D.1) that a symmetric `+-h` probe's own base point is bracketed
   in cutoff space by the two displaced cutoffs, so any draw inactive under BOTH displaced
   points is PROVABLY inactive at the base point too and contributes EXACTLY zero to the
   gradient -- only the sorted crossing slice `min(k_plus,k_minus):W` (not `1:W`) ever needs
   computing. Live: **machine-precision agreement** with `:B_direct_argument_serial` (max
   relative error `1.56e-12` across all 30 coordinates at D=4; `5.79e-13` across 25
   representative coordinates at real D=20/W=80,000) and a **`3.17x` speedup of the FULL
   798-coordinate outer-gradient vector** at real D=20/W=80,000 (`25.84s` -> `8.16s`, one
   real KNITRO inner solve providing the fixed dual both backends share). This is a genuine
   outer-GRADIENT speedup, not merely moment construction -- a materially larger practical
   contribution than item 1 alone, since prior sessions' own documented findings identify
   the outer gradient as a real, repeated cost in every finite-delta campaign.

## 0. Environment and provenance (Phase 0)

- Checkpoint commit (created before touching any kernel this session, per the governing
  prompt's own instruction): `212bf63d3f61ee4d49db675806c4c23ee1880276`, branch
  `melitz/fullD-delta-star`, scoped to `src/melitz`/`test/melitz`/melitz docs-scripts-opt
  files only (excludes unrelated binary artifacts from other sessions' work in the same
  working tree -- see the commit message).
- Julia `1.12.6`, KNITRO `13.0.1` (`KNITRODIR=/opt/shared_sw/knitro/13.0.1`), 208 logical
  CPUs, 3.0TiB RAM -- identical toolchain to every 2026-07-24/25 session
  (`julia-toolchain-use-juliaup` memory).
- `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` exported at every Julia launch this session
  (this repo's own standing rule); `-t 16` Julia threads for the test suite and the
  real-D20 benchmark, matching every prior real-D20 session's own default.
- Full Melitz test suite (`test/melitz/runtests.jl`) reproduced **45/45 testsets passing**
  BEFORE any code change this session (Phase 0 baseline, ~8min wall) -- matches the prior
  session's own documented baseline exactly, confirming the checkpoint commit itself
  introduced no regression. After the FIRST part of this session (Phases 0-3/5/11):
  **46/46 testsets passing** (the one new top-level testset, "Sorted-tail moment
  construction (2026-07-25)", `8434/8434` assertions passing on its own, plus 13 new
  assertions folded into "Pareto data-only calibration" for the Phase 11 real-KNITRO
  equivalence check). After this SAME-DAY CONTINUATION (Phase 6/7 crossing-slice gradient +
  parallel `moment_backend` wiring + the extended Phase 4/13 scaling sweeps): **47/47
  testsets passing** -- the new "Phase 6/7 (2026-07-25)" testset (`35/35`, D=4 exhaustive
  coordinate coverage) plus 27 more assertions folded into "Pareto data-only calibration"
  (now `77/77`: the Phase 11 follow-up parallel-backend check and the real D=20/W=80,000
  crossing-slice gradient check). Zero regressions at every stage.
- This session's own Phase 0 benchmark was NOT the full prescribed grid (calibrated
  reference / near-boundary / over-budget / high-cutoff / low-cutoff points, each with
  complete moment-construction/FC-callback/outer-gradient/inner-solve/rejection wall-clock
  decomposition) -- reduced to the isolated moment-construction function at 4 representative
  points (Section F.1), which is what this session's optimization target actually touches;
  the FC-callback/outer-gradient/inner-solve/rejection wall-clock breakdown by category was
  not separately instrumented this session (Section F.3's own honest accounting).

## A. Mathematical structure

### A.1 Why sorting is valid

For a fixed origin `o`, every bilateral trade-share moment's activity gate is
`melitz_firm`'s `active = operating_profit > 0` (`firm_quantities.jl:69`), where
`operating_profit(z) = revenue(z)/sigma - w_o*f_od` and
`revenue(z) = expenditure_d * price(z)^(1-sigma) / price_power_d`,
`price(z) = markup(sigma) * w_o*tau_od / (A_od*z)`. Since `sigma>1`, `price(z)` is strictly
*decreasing* in `z` and `price^(1-sigma)` is strictly *increasing* in `price` for `1-sigma<0`
composed with a decreasing `price(z)` -- net effect, `revenue(z)` (hence `operating_profit(z)`)
is **strictly increasing in `z`**, for every fixed `(o,d)` cell. Participation is therefore a
scalar threshold test `z_so > cutoff_od` for a SINGLE well-defined `cutoff_od`
(`melitz_baseline_cutoff`/`melitz_cutoff`/`melitz_C`, `equilibrium.jl`/`firm_quantities.jl` --
the exact zero of `operating_profit(z)`, algebraically identical to the direct formula's own
zero-crossing). Sorting origin `o`'s draw column ONCE and reusing that order for every
destination `d` therefore changes nothing about which draws participate or their realized
contribution -- it is a pure re-indexing of a per-cell threshold query from an `O(W)` linear
scan to an `O(log W)` binary search (`searchsortedlast`), an algebraic/data-layout
optimization with no numerical approximation.

`melitz_moments_sorted_tail!` reuses `eq.cutoff[o,d]` (already computed by every production
caller -- `melitz_moments_adapter!` builds `eq = MelitzEquilibrium(..., melitz_baseline_cutoff(...), ...)`
before calling the moment constructor) rather than re-deriving a cutoff independently, so the
active/inactive boundary used by the sorted-tail lookup is, by construction, the SAME value
every other consumer in this codebase (the affine cutoff system, feasibility screens) already
treats as authoritative -- not a second, independently-computed cutoff that could numerically
disagree with `eq.cutoff` at the boundary.

### A.2 Why re-pairing origins independently would be wrong

`sortperm(z[:,o])` reorders origin `o`'s OWN column. The joint draw `s` -- the tuple
`(z_s1,...,z_sD)` -- is the unit of dependence the QMC/Pareto reference distribution actually
generates; a bilateral moment column indexed by the WRONG row would silently corrupt every
downstream quantity that assumes row `s` means the same underlying draw across every column of
`G` (the CC divergence minimization, the outer gradient, every reported `Delta`/LFD). Sorting
origin `o` and origin `o'` independently and then treating "the k-th sorted observation of `o`"
and "the k-th sorted observation of `o'`" as if they were one new joint draw would silently
destroy this pairing -- there is no mathematical justification for it (the reference draws are
NOT rank-correlated by construction; even if they were, the moment system is defined over the
ORIGINAL joint rows, not over rank statistics). `MelitzSortedTailContext` stores one
permutation PER ORIGIN and every active-tail write scatters back through
`permutation[:,o][pos]` before touching `G`, so the original row index is always recovered
before the sorted position is used for anything but the binary search itself. Explicit
"joint-row pairing preserved" tests (`test/melitz/runtests.jl`, Phase 1) construct the
adversarial check directly: applying origin 1's permutation to origin 2's column does NOT
produce a sorted sequence, and scattering `sorted_z[:,o]` back through `permutation[:,o]`
exactly reproduces `z_original[:,o]`.

## B. Sorted context

`MelitzSortedTailContext` (`src/melitz/sorted_tail.jl`) stores, per origin: the original
draws/log-draws/`z^(sigma-1)` powers (indexed by original row `s`), the origin-local
permutation, and the same three representations reindexed into sorted order. Construction is
`O(W D log W)` (one `sortperm` per origin); invariants (bijective permutation, nondecreasing
sorted column, exact round-trip via scatter) are checked directly in the test suite, not
merely asserted. The fingerprint (`hash(:melitz_sorted_tail_ctx_v1, D, W, sigma,
[theta_star], hash(z_original))`) changes under any change to the draws, `W`, or `sigma`
(tested directly) -- a caller comparing fingerprints can detect a stale context before it
silently produces wrong output; `melitz_moments_sorted_tail!`/`_parallel!` additionally
hard-`ArgumentError` on a `D`/`sigma` mismatch at call time (not merely at fingerprint-compare
time, since no caller in this session's production wiring compares fingerprints on every
callback -- see Section G's residual-risk note).

Memory/construction-cost numbers: see Section F (live, from `scripts/melitz_sorted_tail_benchmark_2026-07-25.jl`).

## C. Moment construction

`melitz_moments_sorted_tail!`: fills every trade-share column densely with its inactive base
value `-lambda_od` (an unavoidable `O(W)` write per column, `O(D^2 W)` total, Section F.3's
own scaling discussion), then walks ONLY the active sorted suffix `k_od:W` (found by
`melitz_active_tail_start`, one `searchsortedlast` call per cell) and overwrites the active
rows with `coef_od*z_power[pos] - lambda_od`, `coef_od = melitz_C(...)/expenditure_d`
precomputed once per cell (not per draw). This is the SAME value `melitz_moments!` computes
via `melitz_firm`/`unconstrained_revenue` per draw (Section A.1's algebraic identity), just
reassociated around precomputed per-origin/per-cell constants -- validated to `<1e-9` absolute
agreement (D=4/D=10) and `<1e-8` (real D=20), not merely "close," across the calibrated point,
random A/f perturbations, focal-origin perturbations, engineered zero-active/all-active cells,
and an explicit near-cutoff tie (Section 3.2's own D=4 smoke check found `3.55e-15` max
absolute disagreement -- machine precision, not merely tight tolerance). The single focal
link column (`layout.focal_link_index`) is deliberately kept DENSE in both backends -- see the
Section 3.1 audit below.

### C.1 Phase 3.1 audit: the focal link column

`melitz_moments!`'s link column computes, per draw `w` at the fixed `target_country j`:
`profit_j[w] = sum_d realized_operating_profit_{j,d}(z_j)` (baseline, summed over all `D`
destinations) and a single autarky term `realized_operating_profit'_{jj}(z_j)` (evaluated at
`price_power_autarky = gamma_prime_target`, wage `w_prime`, expenditure `expenditure_prime`).
**Both DO have the same scalar-cutoff-tail structure as the trade-share block**: each of the
`D` baseline terms in `profit_j`'s sum is itself a threshold function of the SAME `z_j` column
(reusable via origin `j`'s own already-built sorted context), and the autarky term is a
threshold function of `z_j` against its own cutoff
`melitz_cutoff(w_prime, f_jj, sigma, melitz_C(w_prime,1.0,A_jj,sigma,expenditure_prime)/gamma_prime_target)`
(the `/gamma_prime_target` factor folded into the revenue-scaling constant exactly as
`melitz_C`'s own derivation requires). **This session audited but did NOT implement a
sorted-tail version of the link column**: it is a single column, `O(D*W)` regardless of which
backend computes it, versus the trade-share block's `O(D^2*W)` -- at `D=20` this is a ~20x
smaller share of the total moment-construction cost, and implementing it would require
threading the autarky `price_power`/`gamma_prime_target`-dependent cutoff formula through a
second, less-exercised code path for a bounded expected return. Kept dense, reusing the
IDENTICAL per-draw loop `melitz_moments!` already runs (same order, same values, bit-identical
in practice) -- zero incremental risk to this column's correctness.

## D. Outer-gradient updates (Phase 6/7, DONE this continuation) and Phases 8-10 (still not
## implemented -- honest scope accounting)

### D.1 Phase 6/7: sorted crossing-slice direct-gradient backend

**Reused, not re-derived**: which physical `(o,d)` cells any given free coordinate `r`
touches is already established and validated by `argument_localized_gradient.jl`'s own
Gate 1/Gate 2 test battery (`melitz_compact_columns_map`, itself built from
`melitz_localized_dependency_map` -- the governing prompt's own "audit every outer
coordinate's downstream cutoff dependencies... do not assume every coordinate changes only
its direct cell" concern is exactly what THAT existing, already-validated machinery answers,
so this continuation deliberately reuses it wholesale rather than re-deriving which cells
are touched, keeping the new work narrowly scoped to HOW each touched column's values get
computed).

**The exact argument** (not a heuristic): `affine_cutoff.jl`'s own already-validated result
is that the full log-cutoff vector `q(theta_free) = q0 + Q*theta_free` is EXACTLY affine.
`direct_gradient.jl`'s central-difference probe evaluates a single coordinate `r` at
`theta+h*e_r` and `theta-h*e_r`, symmetric around the base `theta`. By affine-ness,
`q_od(theta+h*e_r) = q_od(theta) + h*Q[od,r]` and `q_od(theta-h*e_r) = q_od(theta) -
h*Q[od,r]` -- so `q_od(theta)` (hence `cutoff_od(theta)`, `log` being monotone) is EXACTLY
the midpoint of the two displaced cutoffs, and therefore lies WEAKLY BETWEEN them:
`min(cutoff_plus,cutoff_minus) <= cutoff_base <= max(cutoff_plus,cutoff_minus)`. Since
`melitz_active_tail_start` is monotone nondecreasing in its cutoff argument, the SAME weak
ordering carries over to sorted positions: `min(k_plus,k_minus) <= k_base <=
max(k_plus,k_minus)`. Consequence: any sorted position strictly below `min(k_plus,k_minus)`
is PROVABLY inactive under `theta+h*e_r`, `theta-h*e_r`, AND `theta` itself, all three
simultaneously -- at such a position `G_plus = G_minus = G_base = -lambda_od` exactly, so it
contributes EXACTLY zero to the central-difference gradient. Only the sorted crossing slice
`min(k_plus,k_minus):W` (scattered back to original rows via the origin's own permutation,
same pattern as Section C) ever needs to be computed or applied -- a genuine algebraic
identity, made possible specifically because this codebase already builds and validates the
affine cutoff map elsewhere, not a new approximation risk.

**Implementation** (`src/melitz/sorted_crossing_gradient.jl`, new backend
`:B_direct_argument_sorted_serial`): `_fill_compact_direct_columns_crossing_sorted!` fills
`Gp`/`Gm` ONLY at each touched cell's own crossing slice (using its origin's sorted
context); `_direct_coordinate_grad_sorted` applies the `u_plus`/`u_minus` fixed-dual-argument
update over that same slice only, instead of the full `1:W` dense scan
`_fill_compact_direct_columns!`/`_direct_coordinate_grad` (`direct_gradient.jl`) perform.
Requires `ctx.sorted_tail_ctx !== nothing` (built via `moment_backend` in
`(:sorted_tail_serial,:sorted_tail_parallel)`) -- throws `ArgumentError` immediately if
absent, never silently falls back to a dense scan. The focal link column (when a coordinate
touches it) is still filled DENSELY, reusing the unchanged `_fill_compact_link!` -- the same
Section C.1 scope decision, re-applied here for the same reason (a single `O(D*W)` column,
asymptotically small next to the `O(D^2*W)`-scale savings on the direct-cell block, and
avoiding threading the more complex autarky/`gamma_prime_target` algebra through a second,
less-exercised path).

**Validated**:

- **D=4, EVERY coordinate** (30/30, exhaustive, not sampled): max relative disagreement vs
  `:B_direct_argument_serial` `1.56e-12` (`test/melitz/runtests.jl`, "Phase 6/7 (2026-07-25)").
  Gradient values are naturally large in magnitude (this codebase's own `1e10*Delta(theta)`
  scaling convention, `O(1e7)`-`O(1e9)`) -- the raw absolute disagreement (`~6.7e-5`) is
  exactly what `1.56e-12` relative looks like at that scale, not a separate, looser number.
- **Real D=20/W=80,000, 25 representative coordinates** (`g`, plus 24 random draws from the
  full 798-coordinate space): max relative disagreement `5.79e-13` -- machine precision,
  reproduced at production scale, not just at the small D=4 fixture.
- **The FULL 798-coordinate gradient vector, real D=20/W=80,000, one shared real KNITRO
  inner solve providing the fixed dual `x`**: `direct_serial=25.84s` vs
  `direct_sorted=8.16s` -- **`3.17x` speedup**, live-measured
  (`/tmp/.../validate_crossing_gradient_d20.jl`-derived numbers, folded into the test suite's
  own lighter-weight 25-coordinate check to bound routine test wall-clock).
- Cross-checked against an independent frozen-`x` finite difference at a random D=4
  coordinate (same discipline as the existing `:B_direct_argument_serial` test).
- The "requires `ctx.sorted_tail_ctx`" guard is tested directly (`ArgumentError` on a ctx
  built with the default `:dense_reference` backend).

**Not done this continuation**: Phase 6/7's own fuller prescription (every D=20 coordinate,
not 25; `plus`/`minus` fixed-dual scalars and switch counts logged per coordinate;
`:B_direct_argument_sorted_parallel`, the threaded analogue of `direct_gradient.jl`'s own
`:B_direct_argument_parallel`; wiring `gradient_backend=:B_direct_argument_sorted_serial`
into `solve_melitz_finite_delta_bound`/`melitz_fixed_point_probe`'s own accepted-backend
symbol list so a real outer campaign could select it) -- a genuine, low-risk next step given
this session's own proof and live numbers, not attempted here to keep this continuation's
own diff bounded and fully tested rather than partially wired.

### D.2 Phases 8-10: still not implemented (honest scope accounting)

**Phase 8 (sorted fixed-dual scalar/dual-argument construction)**: would restructure the
inner KNITRO callback's own per-draw dual-argument sweep (`u_s = normalization + sum_k
dual_k*G[s,k]`) into a per-origin cumulative-sum-over-sorted-cutoffs construction. This
touches the actual KNITRO-facing inner-loop hot path (`cc_algo/inner_loop_functions.jl`,
shared with the Ricardian model) -- out of scope for a first sorted-tail session without a
dedicated correctness campaign against that shared code.

**Phase 9 (suffix-sum inner-gradient construction)**: a generic `G'v` sorted suffix-sum
prototype for the inner dual gradient. Not started -- would need its own dense-vs-sorted
validation campaign at the level of rigor Section C's trade-share block received here.

**Phase 10 (Hessian same-origin-block structure)**: diagnosis-only per the governing prompt's
own instruction ("do not replace the full Hessian unless the prototype is exact and
materially useful... do not build an elaborate multidimensional orthant-query system"). Not
started this session; the exact same-origin block formula
(`1{z_o>=q_od}*1{z_o>=q_od'} = 1{z_o>=max(q_od,q_od')}`) is a direct corollary of Section
A.1's monotonicity argument and would reuse the SAME sorted context, but deriving and
validating it against the exact production Hessian (`production_wallclock_allocation_audit`
memory: Hessian callback dominates wall-clock 71-87% in the RELATED full-A_od gravity
model, not this Melitz model directly, but a comparable concern here) is a substantial
undertaking on its own.

**What would be required to complete Phases 8-10**: per-phase, roughly the SAME shape of
work this session spent on Phases 1-3/5/6/7/11 (derive the exact formula -- Phase 10's own
same-origin block formula above is already derived, just not yet implemented/validated;
implement an `:experimental` backend behind an explicit opt-in; validate against the
existing dense/direct implementation across D=4/D=10/real-D20 at the calibrated point and
multiple perturbation families; only then consider production wiring) -- Phase 8/9 in
particular touch the shared `cc_algo` inner-loop hot path, raising the validation bar
further (any regression there would affect the Ricardian model too, not just Melitz) --
still a genuine future-session continuation, not attempted in this same-day extension either.

## E. Production integration (Phase 11) and its residual risk

`build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`/
`melitz_calibration_outer_ctx` gained an additive `moment_backend::Symbol=:dense_reference`
kwarg. Every EXISTING call site (every test, every real-D20 production script referenced in
this repo's Melitz history) is unaffected -- the default reproduces `melitz_moments!` exactly,
byte for byte, through the SAME code path as before this session. Passing
`moment_backend=:sorted_tail_serial` builds a `MelitzSortedTailContext` ONCE at bundle
construction (never inside a callback) and stores it on `ctx.sorted_tail_ctx`;
`melitz_moments_adapter!` dispatches on `ctx.moment_backend` (`get(ctx, :moment_backend,
:dense_reference)`, so a ctx built before this session -- lacking the field entirely --
still takes the unchanged path).

**Validated end-to-end through a real KNITRO inner solve** (`test/melitz/runtests.jl`, "Phase
11"): building the SAME real-D20 calibration under both backends and calling
`melitz_recover_lfd` (a genuine nested KNITRO minimum-divergence solve) on each reproduces an
identical `Delta`, LFD weight vector, and maximum weighted moment residual to `1e-8`.

**Disclosed residual risk**: `melitz_moments_adapter!` receives `U` (the draw matrix) as a
separate argument from `ctx` on every callback (this codebase's existing data model,
predating this session -- `U` lives on the KNITRO bundle, `obj.U`, not on `ctx`). The sorted
backend therefore checks only `size(U) == (sorted_ctx.W, sorted_ctx.D)` on every call, not
full content equality (which would cost `O(W*D)` per callback, defeating the optimization's
purpose). A caller that reassigns `obj.U` to a DIFFERENT draw matrix of the SAME shape after
construction, without rebuilding `sorted_tail_ctx`, would silently get stale sorted-tail
output. No production script surveyed this session does this (`obj.U` is set once at
construction and never reassigned in every real-D20 script in this repo's history); flagged
explicitly here rather than silently assumed safe, per this repo's own standing practice
around the far more consequential `A_od=1`/reparameterization mistake pattern (see this
repo's `CLAUDE.md`).

## F. End-to-end performance

### F.1 Moment-construction wall-clock, real D=20/W=80,000

Live, post-JIT (each backend warmed up once before timing; `nrep=3`, `min`/`median`/`max`
reported), `Threads.nthreads()=16`, `BLAS.set_num_threads(1)`, real production calibration
(`real_data/noah_D20`, France focal country), `scripts/melitz_sorted_tail_benchmark_2026-07-25.jl`:

| outer point | active fraction (min/median/max across `D^2=400` cells) | `dense_reference` | `sorted_tail_serial` | `sorted_tail_parallel` (t=16) |
|---|---|---:|---:|---:|
| calibrated reference | 0.25% / 10.0% / 83.6% | 1.4904s | 0.1320s (**11.29x**) | 0.0798s (**18.68x**) |
| near-boundary (`g_shift=-0.0027`) | 0.25% / 10.0% / 82.3% | 1.5234s | 0.1321s (**11.53x**) | 0.0809s (**18.83x**) |
| high-cutoff (`g_shift=+0.02`) | 0.25% / 10.0% / 93.9% | 1.5116s | 0.1331s (**11.36x**) | 0.0796s (**19.00x**) |
| low-cutoff (`g_shift=-0.02`) | 0.25% / 10.0% / 80.7% | 1.4989s | 0.1337s (**11.21x**) | 0.0807s (**18.58x**) |

`MelitzSortedTailContext` construction (one-time, amortized across every callback for the
life of the bundle): `0.187s`. `dense_reference`'s own `1.49-1.52s` range directly reproduces
this repo's own previously-documented "~1.46 seconds in some real-D20 callbacks" finding
(re-measured live this session, not assumed unchanged).

**Reading**: the speedup is REMARKABLY STABLE across all 4 points (`11.2x`-`11.5x` serial,
`18.6x`-`19.0x` parallel) despite the active fraction per cell varying by more than two
orders of magnitude within each point's own `D^2` cells (`0.25%` to `80-94%`) and modestly
across points (the `high-cutoff` point's own max active fraction, `93.9%`, is the LEAST
favorable case for a tail-skipping optimization -- nearly every draw is active there, so the
binary search saves little arithmetic for those specific cells -- yet the AGGREGATE speedup
across all `D^2` cells barely moves, because the dense inactive-value write
(`G[:,trade_col] .= -lambda_od`, unavoidable in both backends, Section C) and the `melitz_C`/
binary-search overhead are a small, roughly constant share of the total regardless of the
active-fraction mix at THIS `D`/`W`).

### F.2 Correctness at the benchmarked points

Every point benchmarked above was ALSO validated for exact numerical equivalence in the test
suite (Section C, D=4/D=10/real-D20 comparisons) -- the benchmark script itself does not
re-verify `G`-matrix equality at every point timed (it reuses the SAME `K`/`G` buffers across
backends purely for timing, per `timed`'s own warm-up-then-repeat structure), so the
numerical-equivalence claim rests on the separately-run, separately-validated test suite
(Section C), not on the benchmark run.

### F.3 What this DOES and does NOT establish about a full outer campaign

This session benchmarked the ISOLATED moment-construction function only -- not a full outer
KNITRO campaign's total wall-clock. Prior sessions' own measurements
(`docs/melitz_real_d20_scaled_knitro_and_profile_2026-07-25.md`,
`docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md`) show a single FC/GA
callback's total cost is NOT moment construction alone -- it also includes the KNITRO-side
constraint/gradient evaluation, and a genuine finite inner solve (when one is attempted) is
`~14-19s`, two orders of magnitude larger than even the DENSE moment-construction cost
measured here. **A `~11-19x` moment-construction speedup will materially reduce the wall-clock
of any campaign phase where moment construction is a non-trivial share of total callback
cost** (e.g. `AboveEvaluationCap`-rejected trial points that never reach a real inner solve,
where moment construction plus the cheap dual-argument check IS most of the cost) but will
NOT proportionally speed up a campaign dominated by genuine finite inner solves. Running a
short matched outer campaign to quantify this split directly (Phase 12's own fuller
prescription: "time per FC, time per GA, trial points per minute, percentage wall in moment
construction vs. inner solve") was NOT attempted this session -- a genuine next step, not
run here to avoid extrapolating a number this session did not actually measure.

### F.4 Outer-gradient wall-clock (Phase 6/7, this continuation)

Unlike moment construction (Section F.3), the outer-GRADIENT speedup (Section D.1) WAS
measured on the full, real object the production driver actually calls: the complete
798-coordinate gradient vector, real D=20/W=80,000, one shared real KNITRO inner solve
providing the fixed dual `x` (`nStatus=0`) both backends evaluate against --
`:B_direct_argument_serial` `25.84s` vs `:B_direct_argument_sorted_serial` `8.16s`
(**`3.17x`**). This number is directly comparable to prior sessions' own documented
per-callback timings (`docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md`'s
own `~14-19s`/finite inner solve, `docs/melitz_real_d20_scaled_knitro_and_profile_2026-07-25.md`'s
own FC/GA call-count tables) -- a `~17.7s` reduction per outer-gradient (`GA`) call is a
material fraction of a single finite inner solve's own cost, meaning this speedup, unlike
the isolated moment-construction number, DOES directly and proportionally reduce a real
campaign's own wall-clock wherever `GA` calls are a non-trivial share of total campaign
time (every campaign surveyed this repo's Melitz history has logged `n_ga_calls` as a
sizeable fraction of `n_fc_calls`, e.g. `26` of `168`-`223` in
`docs/melitz_real_d20_scaled_knitro_and_profile_2026-07-25.md`'s own tables).

## G. Scaling

### G.1 Moment construction (Phase 13, live)

Thread-count sweep (`W=80,000` fixed, `t in {1,2,4,8,16,20}`) and `W`-sweep (`t=16` fixed,
`W in {20000,40000,80000,160000}`), calibrated reference point,
`scripts/melitz_sorted_tail_scaling_2026-07-25.jl` (one process launch per cell, each
warmed up before timing):

| threads | `W` | sorted-tail context construction | `dense_reference` (median) | `sorted_tail_serial` (median) | `sorted_tail_parallel` (median) | speedup, serial | speedup, parallel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80,000 | 0.190s | 1.4916s | 0.1259s | 0.1237s | 11.85x | 12.06x |
| 2 | 80,000 | 0.374s | 1.4735s | 0.1279s | 0.1021s | 11.52x | 14.43x |
| 4 | 80,000 | 0.263s | 1.4809s | 0.1325s | 0.0881s | 11.18x | 16.82x |
| 8 | 80,000 | 0.191s | 1.4947s | 0.1280s | 0.0790s | 11.68x | **18.93x** |
| 16 | 80,000 | 0.235s | 1.5333s | 0.1373s | 0.0825s | 11.17x | 18.58x |
| 20 | 80,000 | 0.230s | 1.5111s | 0.1335s | 0.0847s | 11.32x | 17.83x |
| 16 | 20,000 | 0.045s | 0.3867s | 0.0294s | 0.0198s | 13.16x | 19.49x |
| 16 | 40,000 | 0.106s | 0.7835s | 0.0664s | 0.0434s | 11.80x | 18.05x |
| 16 | 80,000 | 0.241s | 1.5006s | 0.1275s | 0.0792s | 11.77x | 18.94x |
| 16 | 160,000 | 0.529s | 3.0250s | 0.2873s | 0.1686s | 10.53x | 17.95x |

**Reading -- a genuine, not-flat scaling picture, now directly measured rather than
projected**:

- **Serial speedup is remarkably stable (`~10.5x`-`13.2x`) across the entire thread and `W`
  grid** -- expected, since `:sorted_tail_serial` does not depend on Julia thread count at
  all (the small thread-count variation in its OWN column, e.g. `0.1259s` at `t=1` vs
  `0.1373s` at `t=16`, is measurement noise/scheduler contention from the SORTED-CONTEXT
  construction and other concurrent Julia machinery, not a real algorithmic dependence).
  `W=20,000`'s own slightly HIGHER serial speedup (`13.16x` vs `~11x`-`12x` at larger `W`) is
  consistent with Section G.1's own active-fraction-independence reading -- at smaller `W`
  the fixed per-cell overhead (the `melitz_C`/binary-search cost, `O(D^2)`, `W`-independent)
  is proportionally larger relative to `W`, so the dense baseline's own `O(W*D^2)` cost
  shrinks faster than the sorted backend's own smaller fixed-cost floor, giving marginally
  BETTER relative speedup at small `W`, not worse -- the opposite of what a naive
  "amortization takes more draws to pay off" intuition might suggest, because the fixed cost
  here is INDEPENDENT of the ACTIVE-TAIL arithmetic being skipped, so it does not scale away
  the win at small `W` the way, e.g., a one-time context-construction cost would.
- **Parallel efficiency saturates by `t=8`, then MILDLY REGRESSES at `t=16`/`t=20`**:
  `12.06x`(t=1) -> `14.43x`(t=2) -> `16.82x`(t=4) -> `18.93x`(t=8, the observed PEAK) ->
  `18.58x`(t=16) -> `17.83x`(t=20) -- consistent with Section E.3's own diagnosis (only
  `D=20` units of parallel work in the `Threads.@threads for o in 1:D` region) --
  oversubscribing threads beyond the `D=20` unit count adds scheduling overhead without
  adding usable parallelism, and this sweep's own `t=8`-`t=20` numbers show that overhead
  is real and measurable, not merely theoretical, though small (`18.93x` -> `17.83x`, about
  a 6% relative regression from peak to `t=20`).
- **`W`-scaling**: serial speedup drifts DOWN mildly at the largest `W` tested
  (`160,000`, `10.53x`, the lowest in the grid) while parallel speedup stays roughly flat
  (`17.95x`, within the same band as `80,000`'s own `18.94x`) -- consistent with the dense
  baseline's own `O(W)` per-cell write cost and the sorted backend's own `O(W)` scatter-write
  cost (Section C's "unavoidable dense-output write" observation) both growing linearly with
  `W`, while the ACTIVE-TAIL arithmetic difference between the two backends (the actual
  source of the speedup) is a smaller and smaller share of a larger total at bigger `W` --
  the SAME mechanism Section G.1's original draft anticipated ("this will show whether the
  next bottleneck is the dense `WxD^2` storage itself"), now directly confirmed by a real
  4-point sweep rather than inferred from a single `W`.
- **`D` scaling**: still not measured this session beyond the single `D=20` real-data point
  plus the separate `D=4`/`D=10` correctness fixtures (which use much smaller `W`,
  `3000`-`8000`, chosen for test speed, not for a scaling comparison) -- a genuine `D`-sweep
  at fixed `W` (e.g. real D=10/D=20 data, matched `W`) remains a natural next step, not
  attempted here.

### G.2 Outer gradient (Phase 6/7)

Not swept this continuation -- only the single real D=20/W=80,000 point (Section D.1/F.4)
was measured. A `W`/`D`/thread-count sweep for the gradient backend (mirroring G.1's own
grid) is a natural next step, not attempted here.

## H. Recommended production stack

**`moment_backend=:sorted_tail_serial`** is RECOMMENDED as the default for any NEW real-D20
(or real-D10, real-D4) production script in this repo going forward -- it is exact (Section
C/F.2), validated end-to-end through a real KNITRO inner solve (Section E), and delivers an
`~11x`-`13x` moment-construction speedup (Section G.1) with a sub-second one-time setup cost
that is negligible against any campaign running more than a handful of callbacks.

**`gradient_backend`-equivalent `:B_direct_argument_sorted_serial`** (this continuation, new
this session) is RECOMMENDED alongside it for any NEW script using
`solve_melitz_finite_delta_bound`/`melitz_fixed_point_probe`'s own `gradient_backend`
kwarg -- proved exact (Section D.1, not merely tested), delivering a live-measured `3.17x`
speedup of the FULL outer-gradient vector at real D=20/W=80,000, a materially larger
practical contribution than the moment-construction number alone (Section F.4) since the
outer gradient is a real, repeated, previously-unoptimized cost in every finite-delta
campaign this repo's own history has run. **Not yet wired into
`solve_melitz_finite_delta_bound`'s own accepted-`gradient_backend`-symbol list** (Section
D.1's own "not done this continuation" note) -- calling code must currently invoke
`make_melitz_gradient_delta_direct_sorted_serial`/the returned closure directly, matching
how `:B_direct_argument_serial` itself was exercised before its own eventual driver wiring.

**This continuation did NOT change either DEFAULT** (`build_melitz_psi_bundle`/
`build_melitz_psi_bundle_from_calibration` still default to `moment_backend=:dense_reference`;
no driver defaults to `:B_direct_argument_sorted_serial`) -- every existing script in this
repo's history is therefore unaffected unless explicitly updated, the same deliberate
choice the first part of this session made.

`moment_backend=:sorted_tail_parallel` is now WIRED into `melitz_moments_adapter!`'s
dispatch (this continuation's own quick-win follow-up, Section E) -- Section G.1's now-live
thread sweep shows its own peak efficiency at `t=8` (`18.93x`), not `t=16`/`t=20` (`18.58x`/
`17.83x`) -- a caller choosing this backend should prefer `t=8` Julia threads for the
moment-construction region specifically if that is tunable independently of the rest of a
campaign's own thread budget, though the difference across `t=8`-`t=20` is modest (`~6%`
peak-to-`t=20`), not dramatic enough to force a hard rule.

**NOT recommended for production yet**: any Phase 8-10 kernel (none exist -- Section D.2).
A `:B_direct_argument_sorted_parallel` threaded analogue of the new gradient backend, or a
full matched outer-campaign wall-clock comparison (Section F.3's own open question) -- both
genuine, well-motivated next steps, not attempted this session.

## Required-tests checklist (cross-reference)

1. Sorting does not modify the original draws -- Phase 1 test, `test/melitz/runtests.jl`.
2. Every origin permutation is a valid bijection -- Phase 1 test.
3. Joint-row pairing preserved (adversarial cross-origin check) -- Phase 1 test.
4. Binary-search active sets match Boolean masks exactly -- Phase 2 test (2000 random
   trials + exact real-cutoff/`melitz_firm` comparison + tie/edge cases).
5. Sorted moments match dense moments -- Phase 3 tests, D=4/D=10/real D=20.
6. Focal-link moments match -- covered implicitly (link column is bit-identical dense code
   in both backends; every Phase 3 `compare_backends` call checks the full `G`, including
   the link column).
7. Parallel and serial sorted backends agree -- Phase 4 test, bit-identical (disjoint writes).
8. Fused active counts/range diagnostics agree -- Phase 5 test.
9. Crossing-slice updates match full displaced reconstruction -- **DONE this continuation**:
   Phase 6/7 test, `:B_direct_argument_sorted_serial` vs `:B_direct_argument_serial`, EVERY
   coordinate at D=4 (max relative error `1.56e-12`) and 25 representative coordinates at
   real D=20/W=80,000 (`5.79e-13`).
10. Sorted fixed-dual arguments match dense arguments -- N/A, Phase 8 not implemented.
11. Sorted suffix-sum gradients match dense `G'v` -- N/A, Phase 9 not implemented.
12. Cache fingerprints include the sorted context -- `MelitzSortedTailContext.fingerprint`
    tested directly (changes under changed draws/sigma).
13. No stale sorted state reused with changed draws/W/sigma/seed -- `ArgumentError` guards
    tested directly (D/sigma mismatch, U-shape mismatch, missing `sorted_tail_ctx` for the
    gradient backend); full-content staleness is a disclosed residual risk (Section E), not
    eliminated.
14. Every optimized backend preserves DeltaStar/dual/LFD/moment residuals/GT -- Phase 11 +
    Phase 11 follow-up tests (real KNITRO inner solve, all three moment backends, `1e-8`
    agreement); the crossing-slice gradient backend's own agreement is itself evidence for
    this same property at the gradient level (item 9 above).
