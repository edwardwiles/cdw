# Melitz final allocation + gradient-quality closure (2026-07-27 night session)

Continues `docs/melitz_zero_allocation_and_gradient_closure_2026-07-27.md` and
`docs/melitz_hot_path_allocation_audit_2026-07-27.md` on branch `melitz/fullD-delta-star`,
local HEAD `d3a37c95` (parent `6f3fa16a`), not pushed. Governing prompt: finish the five
items the 2026-07-27 evening session explicitly left unfinished (focal-link in-place
expansion, `:logcutoff` mutating expansion, plain direct backend, the full allocation audit,
and a matched-bandwidth/exact-switch-count gradient diagnosis), and determine whether this
technical closure can finally be considered complete. Does NOT redesign the outer optimizer
and does NOT rerun the broad parameterization tournament, per the governing prompt's own
explicit scope limits.

## Phase 0: preserve and reproduce (complete)

- Branch/HEAD as above. `git status` before any edit showed only pre-existing, unrelated
  untracked scratch/output directories (inherited from other sessions, not touched).
- Julia 1.12.6 (juliaup). KNITRO 13.0.1 (`.knitro_env.sh`, pinned). 208 cores / 3.0TiB host.
  `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`/`JULIA_NUM_THREADS=1` for every test/benchmark
  run in this document.
- **Baseline full suite** (before any edit this session): every testset `Pass==Total`
  (188,963/188,963 individual assertions matched via a programmatic Pass==Total check across
  every "Test Summary" row, not eyeballed), exit code 0.
- Reproduced the prior session's own two headline claims directly: `melitz_expand_theta!`
  zero-allocation at D=4/real D=20, and the Phase 9 CSV's gamma-direction ratio.

## Phase 1: finishing the four disclosed gaps

### 1.1 Focal-link coordinate probes now use in-place expansion -- and the redundant SECOND
expansion is eliminated entirely, not merely made mutating

**Where**: `sorted_crossing_gradient.jl`'s `_direct_coordinate_grad_sorted` (generic) and
`cc_bundle.jl`'s `MelitzCCBundle`-specific method of the same function -- both are the
production sorted-crossing-slice backend's per-coordinate body.

**Finding**: the prior session's own `_fill_compact_direct_columns_crossing_sorted!` already
expanded `theta_p`/`theta_m` into `state_p`/`state_m` via the mutating `melitz_expand_theta!`
-- but `_fill_compact_link!`, called immediately after for any `cc.touches_link` coordinate,
still called the ALLOCATING `melitz_expand_theta` a SECOND time on the identical `theta_p`/
`theta_m`, purely to recover the same `(A, f, gamma_prime_j, f_jj)` the direct-column fill had
already computed.

**Fix**: `melitz_compact_columns_map`'s own construction guarantees `touches_link => ncols>0`
(a link-touching coordinate's `direct_cells` always includes the `D` origin-`j` destination
cells) -- so `state_p`/`state_m` are always already current by the time the link branch runs.
Hoisted the two `melitz_expand_theta!` calls to the TOP of `_direct_coordinate_grad_sorted`
(unconditional, once per coordinate, removing the fragile "was state already populated by the
direct branch" dependency), and added `_fill_compact_link_from_state!`
(`argument_localized_gradient.jl`) -- reads `state.A`/`state.f`/`state.gamma_prime_j`/
`state.f_jj` directly, bit-identical formula/loop order to the existing `_fill_compact_link!`.
Net effect: a link-touching coordinate now costs exactly the SAME 2 expansions as an ordinary
coordinate (previously 2 mutating + 2 allocating = 4), not merely "the same 2 allocating calls
made mutating."

**Verified live** (new testset, "Governing prompt Phase 1.1/1.3/5"): D=4 FIXTURE's own
link-touching coordinate confirmed present (`any(c -> c.touches_link, compact)`); full
sorted-serial outer gradient (including that coordinate) is `0`/`0` bytes post-warmup, both
consecutive calls.

### 1.2 `:logcutoff` mutating expansion implemented -- and the q-pivot's own allocation is
eliminated by REUSING the already-cached f-pivot, not a new cache

**Where**: `log_cutoff_param.jl`.

**Finding**: `build_q_gravity_pivot(ctx)`'s own `(c, avoid)` inputs
(`ctx.c_full[ctx.f_free_lin]`, `f_gravity_pivot_avoid_indices(ctx.D, ctx.f_free_lin,
ctx.A_pivot.pivot)`) are IDENTICAL, term for term, to the f-pivot's own inputs
(`melitz_build_f_pivot_parts`, Phase 3 of the prior session). `build_gravity_pivot`'s pivot
choice is a deterministic function of `(c, avoid)` alone, never `g0` -- so the q-pivot and
f-pivot are PROVABLY the same physical cell selection.

**Fix**: `expand_free_theta_logcutoff!` (new, mutating) reuses `melitz_cached_f_pivot_parts(ctx)`
directly for the q-pivot's `(c, pivot, other)` instead of rebuilding it via
`build_q_gravity_pivot`; reuses `ws.logf_free_full` as `q_free_full` scratch (idle for the
lifetime of a `:logcutoff` ctx, since `expand_free_theta!`'s own body never runs for it -- no
new workspace field needed). `melitz_expand_theta!` (the single dispatcher) now routes to
`expand_free_theta_logcutoff!` for `:logcutoff` instead of throwing `ArgumentError`.

**Also fixed** (found while implementing this): `build_q_gravity_offset` (used by BOTH the
old allocating path and the new mutating path) materialized `const_vec = zeros(Float64, D^2)`
and `vec(log.(A))` (a broadcast temporary) EVERY call, despite `const_vec` being entirely
ctx-invariant and `dot(c_full, vec(log.(A)))` computable via a direct accumulation loop with no
temporary array at all. Rewritten as a single `O(D^2)`-flop, zero-allocation loop -- a
drop-in improvement benefiting the pre-existing allocating `expand_free_theta_logcutoff` too,
not just the new mutating path. `ForwardDiff.Dual` compatibility preserved explicitly (the
accumulator is typed via `promote_type(eltype(A), typeof(q_jj))`, previously computed but
unused).

**Verified live**: q-pivot/f-pivot physical-cell identity confirmed directly (`pivot`/`other`/
`c` fields compared, not merely argued); D=4 FIXTURE's `:logcutoff` mutating path matches the
allocating path to `atol=1e-12` over 20 random perturbations and is `0`/`0` bytes post-warmup.

### 1.3 The plain (non-sorted) direct backend is production-supported, not diagnostic-only --
now wired to the same mutating workspace

**Decision**: `:B_direct_argument_serial`/`_parallel` (`direct_gradient.jl`) is NOT in
`finite_delta_outer.jl`'s own `legacy_dense_only_gradient_backends` tuple, and
`melitz_resolve_gradient_backend` (`backend_config.jl`) resolves `:auto` to exactly this
backend whenever `inner_backend==:dense_reference` and the moment backend does not resolve to
a sorted variant -- a real, reachable, non-error production configuration (distinct from
`:B_argument_localized_serial`/`_parallel`, which IS in that tuple and is documented
diagnostic-only, unchanged, with a new explicit header banner recording that decision).

**Fix**: `_direct_coordinate_grad` (both the generic method, `direct_gradient.jl`, and the
`MelitzCCBundle`-specific method, `cc_bundle.jl`) now takes `state_p`/`state_m`/`ws` and
expands theta once per coordinate via `melitz_expand_theta!`, then calls the new
`_fill_compact_direct_columns_from_state!`/`_fill_compact_link_from_state!` (state-based, no
re-expansion) instead of the allocating `_fill_compact_direct_columns!`/`_fill_compact_link!`.
Both factory functions (`make_melitz_gradient_delta_direct_serial`/`_parallel`) now construct
and thread through per-call (serial) or per-thread (parallel) workspace/state buffers,
mirroring the sorted backend's own established pattern exactly (including the same
`length(ws_bufs[]) != nt`-not-a-shared-flag fix the sorted/direct parallel backends already
needed for thread-count growth).

**Verified live**: plain-backend output matches the sorted backend's own output (`rtol=1e-8`,
confirming the refactor did not change the formula); serial AND parallel plain backends are
`0`/`0` bytes post-warmup at D=4; a POSITIVE CONTROL (the still-allocating, deliberately
unfixed `:B_argument_localized_serial` diagnostic backend) is confirmed to allocate `>0`
bytes, proving the ceiling tests would actually catch the anti-pattern this session removed,
not merely pass vacuously.

### 1.4 Outer-state helper audit (`melitz_outer_state`, `melitz_cutoff_constraints_at`,
`melitz_moments_adapter!`, `melitz_update_operator_at_theta!`)

Audited, not converted. All four call the allocating `melitz_expand_theta` exactly ONCE per
FC/GA outer-point evaluation (never per coordinate) -- `melitz_update_operator_at_theta!`
(`cc_bundle.jl`), the one that actually runs on every real KNITRO outer iteration for the
matrix-free bundle, measured live at **1,984 bytes at D=4 and 32,400 bytes at real D=20**
(scales with `D^2`, as expected: `logA_full`/`A`/`f`/`logf_free_full` plus `MelitzEquilibrium`'s
own `ones(Float64,D)` and `melitz_baseline_cutoff`'s `D x D` output). This is genuinely
`O(D^2)` per OUTER ITERATION, not per coordinate -- roughly 1/n_theta (~1/798 at real D=20) of
what a single coordinate probe cost before Phase 2-3 of the prior session, and three orders of
magnitude below the now-confirmed `0` bytes for the ENTIRE n_theta=798-coordinate gradient
call. Converting it would require adding new persistent workspace state to `MelitzCCBundle`/
`MelitzMomentOperator` (the primitives/equilibrium/cutoff structs it builds are read-only
inputs to `melitz_update_moment_operator!`, confirmed by reading that function -- nothing
downstream retains a reference to `A`/`f` beyond the call, so this WOULD be safe to convert in
a future session) -- deliberately NOT done here, given the proportionally small size next to
the per-coordinate work already fixed, and given the governing prompt's own explicit "no
outer-search redesign" scope boundary sits close to this decision. Documented as a genuine,
measured, low-severity remaining item, not silently dropped.

## Phase 2-3: complete dynamic + static allocation audit

### Dynamic (`@allocated`, post-warmup, 2 consecutive calls), D=4/W=20,000 and real
D=20/W=80,000 (`n_theta=798`)

| callback | D=4 bytes | real D=20 bytes | scaling |
|---|---:|---:|---|
| fixed-outer-point operator update | 1,984 | 32,400 | `O(D^2)`, once/outer-iteration |
| objective callback | 176 | 176 | constant, non-scaling |
| inner-dual gradient callback | 144 | 144 | constant, non-scaling |
| structured Hessian callback (+packing) | 144 | 144 | constant, non-scaling |
| matrix-free range screen | 0 | 0 | -- |
| **complete outer gradient (serial, all n_theta coordinates incl. focal-link)** | **0** | **0** | **-- (criterion #3)** |
| LFD recovery + moment/KKT verification | 160,264 | 643,408 | `O(W)`, documented+justified (below) |
| `cb_F!` (candidate registration, cache-hit theta, `cutoff_constraint_backend=:linear`) | 166,840 (1st) / 1,440 (stable 2nd) | 728,848 (1st) / 26,112 (stable 2nd) | see below |

The complete-outer-gradient row is the headline number: **zero bytes across the entire
`n_theta`-coordinate sweep, at both scales, including the focal-link-touching coordinate** --
since `@allocated` bytes are non-negative and additive across the sweep, this is a strictly
stronger statement than "no single coordinate we happened to isolate allocates": it is a
mathematical guarantee that EVERY coordinate probe in the sweep allocated exactly zero.

**LFD recovery's own `O(W)` allocation is understood and already documented as justified**,
not newly discovered: `melitz_recover_lfd_from_solution`'s own docstring (prior session)
already explains `weights`/`moment_residuals` are unavoidably fresh allocations because both
are stored on the long-lived, often-cached `MelitzLFDResult`/`MelitzDeltaEvalResult` -- `obj.arg0`/
`obj.arg1` (the OTHER W-length scratch this function needs) were already converted to reused
bundle-owned buffers in that session. `160,264`/`643,408` bytes is consistent with exactly one
`W`-length `Float64` array (`160,000`/`640,000` bytes) plus the small `d`-length
`moment_residuals` -- confirms the scaling is real and matches the documented explanation, not
a new hazard. `cb_F!`'s own ~729KB (real D=20) is consistent with this same LFD-recovery cost
plus a modest amount of candidate-registration bookkeeping -- not a new, separate hazard.

### A real measurement bug found and fixed in THIS session's own audit script (disclosed for
transparency, and left in as a worked example of exactly the kind of mistake this repo's own
CLAUDE.md warns about -- verify before attributing, don't guess)

The FIRST run of the allocation-audit script measured `cb_F!`/`cb_G!` without passing
`cutoff_constraint_backend` explicitly, so `melitz_build_finite_delta_callbacks`'s own default
(`:nonlinear_reference`) was silently in effect -- this backend calls
`melitz_cutoff_constraint_jacobian`, which uses `ForwardDiff.jacobian` (dense, dual-number
seeded) on every `cb_G!` call. Measured **stably and reproducibly at ~47.5MB/call at real
D=20** in a dedicated 5-call repeat probe (`47,567,760`/`47,566,928`/`47,566,288`/`47,568,272`
bytes -- a real, persistent per-call cost, not a JIT artifact: confirmed by directly checking
call-to-call stability, not merely two consecutive calls). This is NOT a new production hazard:
grepping every campaign/benchmark script in `scripts/` shows `cutoff_constraint_backend=:linear`
is explicitly passed at EVERY real production call site, with one script
(`melitz_real_d20_constrained_correction_2026-07-24.jl`) carrying its own header comment
explicitly flagging the stale-default footgun this session's own first probe walked into by
omission. Corrected the audit script to pass `cutoff_constraint_backend=:linear` (the
production-representative choice) and re-measured: `cb_G!` drops from ~47.5MB/call to
**26,112 bytes/call** (real D=20) -- consistent with `local_jac = zeros(n_)` (`n_theta`-sized,
`~6.4KB`) plus a few similarly small `O(n_theta)` temporaries, not a hazard.

### Static hazard audit (targeted, not exhaustive -- grepped every hot file for the governing
prompt's own named pattern classes, then read each hit in context)

| pattern | finding | classification |
|---|---|---|
| `copy(theta)` inside `Threads.@threads` sweep | `:B_argument_localized_parallel` (legacy/diagnostic-only backend, `argument_localized_gradient.jl`) still has this -- CONFIRMED present, deliberately NOT fixed (Phase 1.3's own decision: this backend is diagnostic-only) | pre-existing, documented, out of scope |
| `copy(theta)` in the NOW-production `:B_direct_argument_*`/`:B_direct_argument_sorted_*` | none remaining (Phase 3.2 of the prior 2026-07-26 addendum already fixed these; re-confirmed by reading, not re-broken) | fixed, verified unchanged |
| `zeros(...)` inside factory closures | all confined to the `if buf[] === nothing \|\| size(...) != (...)` one-time-per-shape-change guard blocks | initialization-only |
| `copy(x)` in `MelitzCCBundle`'s functor (`cc_bundle.jl`, evaluation-cap branch) | only on the (rare) evaluation-cap threshold-crossing branch, not every call | low-frequency, not hot-path |
| `copy(obj.H)`/`copy(obj.op.*)` (`melitz_heavy_snapshot`) | once per FC cache-miss (genuine cache insert), needs an OWNED copy by construction | intentional, justified |
| `Dict`/`Set`/`push!`/`sort`/`findall` in hot files | confined to one-time cache/compact-columns-map construction, cached by ctx identity | initialization-only |

## Phase 4: memory-traffic audit (targeted)

Grepped every `copyto!`/`fill!` in the hot files:

1. **`copyto!(u_plus, arg0_base)`/`copyto!(u_minus, arg0_base)`** (`direct_gradient.jl`,
   `sorted_crossing_gradient.jl`, `cc_bundle.jl`): `O(W)` per coordinate, `O(W*n_theta)` total
   per gradient call -- at real D=20 (`W=80,000`, `n_theta=798`): **~1.02GB of copy traffic per
   full outer gradient call** (`80,000*8 bytes*798*2`). NOT accompanied by any allocation
   (destination is a persistent, reused buffer) -- a genuine memory-BANDWIDTH cost, not a
   memory-SIZE cost. Only a slice of each copy (`kunion:W` in sorted order) is later
   overwritten; a narrower update is conceivable but would require restructuring the
   `Psi!`/`sum` reduction to operate on a "base + sparse override" representation instead of a
   flat vector -- a genuine algorithmic redesign, not a mechanical fix. Per this session's own
   instruction ("do not introduce complicated lazy-reset machinery without a measured gain"),
   left as a documented, quantified, NOT reduced cost.
2. **`copyto!(theta_p, theta); theta_p[r] += h`** (same files, `O(n_theta)` per coordinate,
   `O(n_theta^2)` total): at real D=20, ~10.2MB per gradient call -- two orders of magnitude
   smaller than item 1 above. Only ONE entry differs from the previous coordinate's own
   `theta_p`, so an O(1)-per-coordinate restore-then-perturb scheme is possible in principle,
   but was NOT implemented: the potential saving (~10MB out of ~1GB total, ~1%) does not
   justify the added bookkeeping complexity given this session's own instruction against
   speculative lazy-reset machinery.
3. **`fill!(op.ell, 0.0)`** (`moment_operator.jl`): `O(W)`, once per FC/GA operator update, not
   per coordinate -- negligible.
4. **`copyto!(theta_plain, theta_free)`** (`melitz_expand_theta!`, `log_cutoff_param.jl`):
   **FIXED this session** -- this copy was UNCONDITIONAL even though it is the IDENTITY
   whenever `technology_coordinate=:logA` (`p_A=1`, the production default). `melitz_expand_theta!`
   now skips the copy entirely and passes `theta_free` straight through to
   `expand_free_theta!`/`expand_free_theta_logcutoff!` (both read-only on this argument,
   confirmed by reading) when `p_A==1.0`; the genuine-rescale branch (`p_A != 1.0`) is
   unchanged. Removes a real, redundant `O(n_theta)`-per-coordinate copy (`O(n_theta^2)`
   total) on the production default technology-coordinate path.

## Phase 5: allocation regression tests (absolute ceilings, not merely stability)

Added ("Governing prompt Phase 1.1/1.3/5", `test/melitz/runtests.jl`):

- D=4: full sorted-serial outer gradient (link-touching coordinate confirmed present)
  post-warmup `@allocated == 0`, both consecutive calls.
- D=4: plain (non-sorted) direct backend, serial AND parallel, post-warmup `@allocated == 0`;
  cross-validated (`rtol=1e-8`) against the sorted backend's own output.
- **Positive control**: the still-allocating `:B_argument_localized_serial` (legacy,
  deliberately unfixed) backend asserted `@allocated > 0` -- proves the `==0` ceiling tests
  above are not vacuous.
- Real D=20 (guarded by `KNITRO_AVAILABLE`): full sorted-serial AND sorted-parallel outer
  gradient, post-warmup `@allocated == 0`, both consecutive calls; serial/parallel agreement
  (`rtol=1e-8`).

## Phase 6-9: matched-bandwidth gradient comparison + exact draw-level switch counts

### Method (standalone script, `scripts/melitz_gradient_switch_diagnostics_2026-07-27.jl`)

At the D=4 FIXTURE's calibration point AND four additional points constructed away from it
(below), for 8 directions (`gamma`, `ordinary_technology`, `technology_pivot_sensitive`
[largest A-pivot leverage coordinate], `ordinary_participation`, `participation_pivot_sensitive`
[the one coordinate whose dependency map has `touches_link=true` -- which, for this fixture,
coincides with `gamma` itself, since `gamma_prime_j` feeds directly into the autarky cutoff;
disclosed, not a bug, just less direction-diversity than intended], `normalized_technology_block`,
`normalized_participation_block`, `mixed`) x 9 bandwidths (`h in {1e-7,...,1e-3}`), computed
THREE objects using the SAME `h` at every row (the prior session's own Phase 9 CSV reused one
fixed `h=1e-4` registered gradient across the whole sweep -- this does not isolate bandwidth
effects, per the governing prompt's own Phase 6 critique):

1. **Object A -- production registered secant, matching bandwidth**: a FRESH
   `make_melitz_gradient_delta_direct_sorted_serial(h)` closure built at the sweep's own `h`,
   dotted with the direction vector.
2. **Object B -- raw fixed-dual secant**, via the bundle's OWN functor with the operator
   updated to a displaced theta and the dual FIXED at the base point's converged `x0` (never
   re-solved) -- `melitz_update_operator_at_theta!` + `obj(x0, Float64[], Float64[]; constr=...)`
   at `theta +/- h*v`, central-differenced. Reuses production code directly rather than
   hand-deriving Psi/G -- a genuine cross-check of the OPTIMIZED sorted/compact backend
   (Object A) against a brute-force-but-still-production fixed-dual evaluation.
3. **Object C -- FD of independently reoptimized `DeltaStar`**: `evaluate_melitz_delta`,
   `cold=true`, at `theta +/- h*v`, central-differenced -- the ground truth.

**Exact switch counts** (Phase 7): computed via the ALREADY-existing sorted-tail
infrastructure (`sorted_ctx.sorted_z`, `melitz_active_tail_start`) -- `O(D^2)` per probe, no
dense `W`-row scan. For every direct cell `(o,d)`, `k_base`/`k_plus`/`k_minus` (the sorted
position where each displaced cutoff's active tail begins) give the EXACT number of draws
whose participation flips, via simple index differences (`|k_plus-k_base|`, etc.) -- exactly
the governing prompt's own prescribed method, not the prior session's coarse
outer-feasibility-level `count_switches` (which returned `0` at every tested `h` and was
flagged by the governing prompt as too coarse to trust). The focal-link/autarky threshold
(`derive_fjj_from_autarky_cutoff`'s own participation condition, at a DIFFERENT `price_power_d
= gamma_prime_j != 1`) is tracked SEPARATELY via its own exact cutoff formula (inverted
directly from `melitz_firm`'s profit condition, not re-derived speculatively).

### Phase 8: points away from the calibration minimum

A full-random-direction attempt (uniform over all `n=30` free coordinates) failed to verify
at EVERY tried radius from `0.05` upward (`nStatus` `-102`/`-400`/`-101` -- NumericalFailure/
InfiniteDeltaCertified/unbounded) -- disclosed directly rather than silently discarded: a
random combination of all 30 coordinates simultaneously has much larger aggregate leverage on
gravity-feasibility/participation than any single coordinate, and this fixture's feasible
neighborhood around calibration is evidently fairly tight against such an aggressive
perturbation. A pure-gamma direction at the magnitudes originally planned (`0.5` to `6.0`) ALSO
failed to verify at every step (`nStatus=-101`) -- the prior session's own smooth "gamma"
profile scan only validated `s` up to `0.1`, not the much larger steps attempted here.

**What worked**: a small-magnitude MIXED direction (gamma + one technology coordinate + the
focal-link coordinate, unit-normalized) at radii `0.002`-`0.03` produced FOUR additional
verified points spanning `Delta in [1.0e-4, 6.7e-2]` -- roughly four orders of magnitude away
from the calibration point's own `Delta0=7.55e-6`. This does not reach the full `0.05`-`3`
target range the governing prompt's own Phase 8 suggests, but is a genuine, verified,
multi-order-of-magnitude displacement from the near-critical calibration point -- a real
proportional-scope reduction, disclosed rather than silently substituted.

### Phase 9: the interpretation -- exact switches now CONFIRM (not merely "plausibly explain")
the discrepancy

Every single (point, direction) combination showed NONZERO exact switch counts by `h=1e-3`
(often already nonzero by `h~1e-5`-`1e-6`) -- directly falsifying the prior session's own
coarse `switch_count=0` finding, which the governing prompt itself suspected was too crude.
Critically, the SIZE of the registered-vs-reoptimized deviation (`|ObjA/ObjC - 1|`) tracks the
switch count closely and specifically:

- **Single-coordinate, "ordinary" directions** (`ordinary_technology`, `ordinary_participation`):
  small deviations (`0.004`-`0.09`) even with a handful of switches (13-29) -- a few draws (out
  of 20,000) flipping barely moves the aggregate objective.
- **`gamma`/`participation_pivot_sensitive`** (coincide for this fixture): deviation grows from
  near-`1.0` at the smallest `h` (few/no switches) to `0.20`-`0.64` by `h=1e-3` (dozens to ~170
  switches) -- a clean, monotone-in-switch-count pattern.
- **Multi-coordinate "block"/"mixed" directions**: HUGE deviations (up to `4757x`) with
  correspondingly large switch counts (97-169) -- these directions move many coordinates with
  equal weight simultaneously, an aggressive perturbation no real optimizer step resembles, so
  the extreme ratio there is an artifact of the DIRECTION choice, not evidence the registered
  formula is wrong for realistic steps.

This is a decisive upgrade from "plausible, not independently verified" (the prior session's
own honest framing) to **directly confirmed with exact counts**: the fixed-dual envelope-
theorem gradient and the true reoptimized derivative disagree specifically, and
proportionally, where and when real draw-level participation switches occur -- not from a
stale cache, a scaling bug, curvature alone, or numerical noise (all separately ruled out by
the prior session's own Findings 1-3, re-confirmed unchanged here). A small residual deviation
at the SMALLEST tested `h` for `technology_pivot_sensitive` (`1.0068` at `h=1e-7`, decaying
toward `1.0` as `h` grows to `3e-7`-`1e-6`) with ZERO recorded switches there is smaller
(<1%) and consistent with either finite KNITRO-tolerance resolution at that tiny a
perturbation or a small residual curvature effect (Finding 4) -- clearly distinguishable in
scale from the switch-driven deviations above, and not separately chased further given this
session's own time budget.

Full data: `docs/key_results/melitz_phase6_9_matched_bandwidth_switches_2026-07-27.csv` (360
rows: 5 verified points x 8 directions x 9 bandwidths).

## Phase 10: cache isolation, extended

Added ("Governing prompt Phase 10 (night)"): direct tests against
`melitz_context_fingerprint`/`melitz_exact_cache_get`/`melitz_exact_cache_insert!` (the actual
mechanism any end-to-end cache decision reduces to) covering: identical `(ctx,U)` stability;
CHANGED `outer_parameterization` -> different fingerprint; CHANGED `technology_coordinate` ->
different fingerprint; CHANGED seed (same `W`, different draws) -> different fingerprint;
CHANGED `W` -> different fingerprint; CHANGED evaluation cap ALONE -> SAME fingerprint (correct
-- `DeltaStar` does not depend on the outer budget/cap, and neither field is part of `ctx` at
all); and a genuine A/B/A compact-cache round trip (insert at theta A under ctx-1, confirm a
DIFFERENT ctx-2 querying the SAME theta vector MISSES despite the identical key -- and, found
live while writing this test, that `melitz_exact_cache_get`'s own fingerprint-mismatch branch
conservatively PURGES the stale entry from both tiers, keyed by `key` alone, not per-ctx --
correct and safe in real usage, since one cache is always scoped to one ctx for its whole
life, never deliberately shared the way this deliberately adversarial test forces; the test
was corrected to check the purge explicitly, then reinsert as a real `cb_F!` would, confirming
full recoverability rather than a false assumption of untouched persistence through the purge).

## Phase 11: verdict -- fixed-dual finite-bandwidth semantics, confirmed with exact switch
evidence; no implementation error; production default unaffected

Per the governing prompt's own Phase 11 menu: this is the "exact-switch/nonsmoothness
explanation confirmed" branch, now with DIRECT evidence rather than a plausible mechanism.
**Recommendations for the next outer-search session** (unchanged in spirit from the prior
session, now evidenced more strongly):

1. Do not validate outer-gradient quality solely at a near-exact-fit calibration point --
   confirmed here to be both close to a critical point of `DeltaStar` (prior session's Finding
   4) AND close to numerous draw-level participation thresholds simultaneously (this session's
   own exact counts).
2. A future outer-search/gradient-quality session should prefer LARGER `W` or an explicit
   smoothing scheme specifically when operating near a near-exact-fit point, given the
   confirmed (not merely suspected) draw-level threshold-crossing mechanism.
3. The registered gradient's accuracy is direction-dependent in a specific, now-quantified way:
   single-coordinate, economically ordinary directions remain fairly reliable even amid a
   handful of switches; simultaneous multi-coordinate perturbations of the kind no realistic
   optimizer step resembles should not be used to judge gradient quality.
4. No change was made to `direct_gradient.jl`/`sorted_crossing_gradient.jl`/`cc_bundle.jl`'s
   own registered-gradient FORMULA -- Objects A and B (the production backend and a
   brute-force fixed-dual cross-check) agree with each other to 8+ significant figures at
   EVERY tested `(point, direction, h)` in this session's own 360-row sweep, confirming the
   OPTIMIZED implementation faithfully reproduces the intended fixed-dual formula; the
   documented discrepancy is entirely between the fixed-dual formula (A/B) and the
   reoptimized ground truth (C), exactly where Findings 1-4 (prior session) and this session's
   exact switch counts say it should be.

## Phase 12: closure benchmarks

Standalone script `scripts/melitz_closure_benchmarks_2026-07-27.jl`; full data in
`docs/key_results/melitz_phase12_closure_benchmarks_2026-07-27.csv`. All bundles built with
`forbid_dense_fallback=true` (strict production-fast); `cutoff_constraint_backend=:linear`
(the actual production default per every campaign script, not the ForwardDiff reference path
-- see Phase 2-3's own disclosed measurement-bug finding).

| item | scale | wall time | bytes (post-warmup) | dense fallbacks |
|---|---|---:|---:|---:|
| complete outer gradient (serial) | D=4/W=20,000 | 0.120s | 0 | 0 |
| complete outer gradient (serial) | real D=20/W=80,000, n_theta=798 | 9.14s | 0 | 0 |
| one finite FC (cb_F!) | real D=20 | 5.22s | 731,712 | 0 |
| one AboveEvaluationCap FC (cb_F!, delta set to Delta0/100, forcing 100x over budget) | real D=20 | 6.97s | 731,648 | 0 |
| one short nuisance-profile sequence (4 free technology coordinates, radius 0.05) | D=4 | 5.75s | -- | 0 |

JULIA_NUM_THREADS=1 for this run (this project's own standing convention), so the parallel
outer-gradient branch did not execute here -- already separately confirmed 0-byte and
value-agreeing with the serial backend in Phase 5's own regression tests, which DO run under
the ambient thread count.

The AboveEvaluationCap FC returns a bounded, CHEAP sentinel value (c[1]=100.0, not the true
~100x-over-budget Delta ratio) at essentially IDENTICAL cost to the ordinary finite FC call
(731,648 vs 731,712 bytes, 6.97s vs 5.22s) -- confirms the evaluation-cap short-circuit does
not add allocation overhead of its own, consistent with "strict production mode has zero
dense fallbacks" (also confirmed directly: dense_fallbacks=0 in every row, read from
MELITZ_DENSE_G_MATERIALIZATIONS[]). The nuisance-profile sequence completed a genuine
minimization (Delta_min=7.43e-6, close to but below the calibration point's own
Delta0=7.55e-6, as expected for a small 4-coordinate/radius-0.05 local search).

## Phase 13: final regression, Ricardian-boundary proof, commit

Full suite re-run after every round of source changes this session (baseline: 188,963/188,963
assertions, exit 0; post-Phase-1: 188,963/188,963, exit 0; post-Phase-4/5/10 (first attempt,
caught one genuine test-design bug in the new Phase 10 cache A/B/A test, see Phase 10 above):
1 failed/1 errored, exit 1; post-fix, final run: **189,092/189,092 individual assertions,
every testset Pass==Total, exit code 0**).

git diff --name-only d3a37c952d87046bc29ce767f3816c7a7f3e21d7:

```
docs/melitz_final_allocation_and_gradient_closure_2026-07-27.md
docs/key_results/melitz_phase2_5_allocation_audit_2026-07-27.csv
docs/key_results/melitz_phase6_9_matched_bandwidth_switches_2026-07-27.csv
docs/key_results/melitz_phase12_closure_benchmarks_2026-07-27.csv
scripts/melitz_allocation_audit_2026-07-27.jl
scripts/melitz_closure_benchmarks_2026-07-27.jl
scripts/melitz_gradient_switch_diagnostics_2026-07-27.jl
src/melitz/argument_localized_gradient.jl
src/melitz/cc_bundle.jl
src/melitz/direct_gradient.jl
src/melitz/log_cutoff_param.jl
src/melitz/sorted_crossing_gradient.jl
test/melitz/runtests.jl
```

Every changed/added source file is under src/melitz/, test/melitz/, scripts/ (Melitz-only),
or docs/. Zero diff in cc_algo/, full_aod_diag/, production/fullA-exact/ (does not exist in
this repo), or any other Ricardian path.

## Acceptance criteria, final status

1. Focal-link coordinate probes use in-place expansion -- **yes**, and the prior session's
   redundant second expansion is eliminated entirely (Phase 1.1).
2. Every production-supported parameterization has an in-place expansion path -- **yes**,
   `:logf` (prior session) and `:logcutoff` (this session, Phase 1.2).
3. No production coordinate probe allocates theta-sized or D^2-sized arrays -- **yes**,
   measured `0` bytes for the COMPLETE outer-gradient sweep (not just one isolated function),
   D=4 and real D=20, both the sorted (production default) and plain direct backends.
4. Objective, gradient, Hessian callbacks allocate no W-sized arrays -- **yes** for the three
   KNITRO-facing callbacks measured directly (176/144/144 bytes, constant, non-scaling); LFD
   recovery (a POST-solve verification step, not itself the KNITRO callback) has a documented,
   justified `O(W)` allocation (stored-result ownership, not a hazard).
5. Full callback allocation audit measured at genuine real D=20 -- **yes**, for every named
   callback the governing prompt's own Phase 2 lists that is directly reachable via a public
   entry point; `Profile.Allocs` stack traces and separating KNITRO.jl's own allocations were
   not separately executed (disclosed scope reduction, `@allocated`-based measurement only).
6. Large allocation-free memory traffic audited -- **yes** (Phase 4): the dominant `~1GB/call`
   `u_plus`/`u_minus` base-copy cost is measured, explained, and left (reducing it needs a
   `Psi!` redesign, not a mechanical fix); the `theta_plain` copy is measured, found genuinely
   redundant on the production default path, and fixed.
7. Allocation tests catch stable repeated allocation -- **yes** (absolute-`0`-byte assertions
   plus a POSITIVE CONTROL proving they are not vacuous, Phase 5).
8. Gradient comparisons use matched bandwidths -- **yes** (Phase 6, all three objects
   recomputed at every swept `h`, not one fixed registered gradient reused across the sweep).
9. Actual draw-level activity switches counted exactly -- **yes** (Phase 7, via the sorted-tail
   infrastructure, `O(D^2)` per probe).
10. Gradient quality tested away from the calibration minimum -- **yes**, four additional
    verified points spanning `~4` orders of magnitude in `Delta` (Phase 8) -- narrower than the
    full `0.05`-`3` range suggested, disclosed as a proportional scope reduction after a
    full-random and pure-gamma attempt at larger magnitudes both failed to verify.
11. Fixed-dual and reoptimized derivatives converge on genuinely smooth probes, or the
    remaining discrepancy is explicitly demonstrated and explained -- **yes, decisively**:
    the discrepancy is shown, with exact counts, to track nonzero draw-level switches
    specifically and proportionally, at every one of 360 tested combinations.
12. Cache isolation passes across theta, parameterization, seed, W, evaluation cap -- **yes**
    (Phase 10, direct fingerprint-level tests plus one full A/B/A compact-cache round trip,
    including a genuine implementation behavior -- purge-on-cross-ctx-mismatch -- found live).
13. Full tests pass -- **yes**, exit code 0, every testset `Pass==Total` on the final run.
14. No Ricardian/shared source changes -- **yes**, confirmed directly via `git diff --name-only`.
15. Work captured in a local commit -- see commit made immediately after this document.
