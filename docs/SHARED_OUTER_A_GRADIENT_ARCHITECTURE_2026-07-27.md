# Shared Outer (A)-Gradient Architecture — 2026-07-27

**Worktree:** `/bbkinghome/edav/gravity_robustness/worktrees/shared-a-gradient-2026-07-27`
**Branch:** `feature/shared-outer-a-gradient-2026-07-27`
**Base commit:** `a69b32d05590120d9d04ada8063178dd7060e1ff` (sibling in-progress operator-stack work)
**This task's commits (in order):** `281e830`, `c5cfddc`, `88aa0c4`, `2781583` (see `git log` in
`key_results/provenance.txt` for the exact hashes/diffstat).

## Verdict block

```text
A_GRADIENT_BACKEND = unrestricted:composite_gradient_at_fast_buffered(default,UNCHANGED_this_task) flexible_cm:legacy_unbuffered(NOT_yet_wired) common_frechet:legacy_unbuffered(NOT_yet_wired) cm_plus_zc:legacy_unbuffered(NOT_yet_wired) zc_only:shared_inplace_pooled(WIRED_DEFAULT_this_task)
A_GRADIENT_ALLOCATION_D20 = unbuffered:9296.14MB(cold)/4474.51MB(warm) buffered:CRASHES(pre-existing DimensionMismatch bug under current :exclude_row default, NOT fixed) pooled_before:CRASHES(same bug) shared_inplace_after:651.24MB(cold)/614.83MB(warm)
A_GRADIENT_SPEEDUP = not separately measured this pass (allocation was the focus; wall-clock numbers are reported in A_GRADIENT_BEFORE_AFTER_PERFORMANCE_2026-07-27.md but are contaminated by machine load -- see that doc's honesty notes)
A_GRADIENT_ALLOCATION_REDUCTION = 93.0% (cold bandwidth cache) / 86.3% (warm bandwidth cache) vs the UNBUFFERED reference at real D=20/W=80,000 -- the only other function that actually RUNS at the current production default (buffered/pooled both crash, see below). At D=4/W=8000: 74.5% (cold) / 25.8% (warm) vs the PRE-EXISTING pooled function.
ALLOCATING_PRICE_CELL_CALLS = 0 (in the shared economic_A_gradient! coordinate loop, both the winner-flip-counting and same-destination-2-origin paths; the rare >2-changed-origin defensive fallback, unreachable for D>=4, still allocates and is disclosed as such)
ALLOCATING_LFIX_INCREMENTAL_CALLS = 0 (a_block_fd_component_ws2!/lfix_incremental_at_ws2! route every reachable-at-D>=4 case through caller-owned GradWorkspace/TwoOriginScratch buffers)
ALLOCATING_LFIX_FROM_Q_CALLS = 0 (lfix_from_q! writes into ws.psi, reused from lfix_buffer_reuse.jl unmodified)
P_WEIGHTS_ARRAY_ALLOCATION = removed (all 5 families: cm_production_bundle.jl, cm_meanzc_production.jl, cm_originzc_production.jl, cm_frechet_cplus.jl, fast_range_screen.jl -- replaced with a verified-bit-identical non-allocating sum)
TYPE_UNSTABLE_HOT_STATES = OriginZCOperatorState (obj::Any, layout::Any, core_cf_ref::Ref{Any}, econ_ws_for::Any) -- NOT touched this pass (task explicitly flags these `Any` fields may be intentional for load-order-safety reasons per a sibling session's own comments; not investigated further given time)
PRODUCTION_MERGE = partial_merge (1 of 5 families wired to the shared backend as its DEFAULT and gated at D=4+D=20; a genuine, previously-undetected correctness bug found and fixed along the way; 4 families NOT wired -- honest gap, see FIVE_FAMILY_SHARED_A_GRADIENT_WIRING_2026-07-27.md)
```

## What this task actually delivered, and what it did not

This was a large task (12 numbered sections). Given the time available and two significant
unplanned detours (a real correctness bug found and fixed, and a second pre-existing bug
discovered and disclosed but not fixed), the honest scorecard is:

| Task section | Status |
|---|---|
| §1 shared `economic_A_gradient!` entry point | DONE — implemented, real evidence (D=4 + D=20 bit-identical vs reference) |
| §2 source-level 756.5MB reconciliation | DONE for D=4 (exact byte-for-byte reconciliation, see below); the cited 756.5MB D=20 figure itself turned out to be **unreproducible under the current production default** (see "A second bug" below) — reconciled qualitatively, not by reproducing that exact number |
| §3 eliminate per-coordinate W-scale allocations | DONE for price/CES cells and the fixed-dual q/Ψ probes (both the single- and 2-origin-same-destination cases); the rare (D<=3, 3+ changed origins) defensive fallback still allocates (disclosed, unreachable for D>=4) |
| §4 persistent L-fix base cache | **NOT ATTEMPTED** — `build_lfix_base_cache` is still rebuilt fresh every gradient call (4.48 MB/call at D=4, the dominant remaining cost in the warm-bandwidth-cache steady state). Flagged as the single highest-value next step; not attempted given the risk (some `LFixBaseCache` scalar fields like `gammafac`/`μ`/`σ` are immutable-struct fields that could legitimately change under flexible-theta, so an in-place refill needs care this session did not have time to give safely) |
| §5 workspace lifecycle | PARTIAL — `EconomicAGradientWorkspace` exists and is NOT rebuilt per gradient call for the one family wired to it; the process-wide `get_or_build_econ_a_grad_ws` cache (keyed by `W`) is a pragmatic approximation of "one per live outer-solver context," disclosed as such, not a true per-context registry |
| §6 precompute coordinate metadata | **NOT ATTEMPTED** — `affected_cells(pe, coord_idx)` is still recomputed inside the coordinate loop on every call (same as every pre-existing gradient function); not touched this pass |
| §7 five-family wiring proof | PARTIAL — 1 of 5 families (ZC-only) wired and proof-gated; the other 4 still call `composite_gradient_at_fast` directly, honestly reported, not claimed otherwise |
| §8 correctness gates | DONE for the shared function itself (D=4 + real D=20, both h_mode) and for the ZC-only wiring (D=4, 3 layouts, full gradient incl. eta block). NOT done for the other 4 families (not wired) |
| §9 allocation/performance gates | DONE at D=4 and real D=20 for unbuffered-vs-shared (see verdict block); buffered/pooled comparison at D=20 is impossible under the current default (crashes) — see below |
| §10 other allocation fixes | p_weights: DONE, all 5 families. Type stability: NOT attempted (time). Hessian: NOT attempted (time, and out of the prioritized order per the task's own deadline guidance) |
| §11 release structure | Followed: 4 separate commits, each independently gated before the next was built on top |
| §12 deliverables | This document set |

## The two bugs found along the way (both real, both disclosed, one fixed)

### Bug 1 (FIXED this task): `dest_contrib_incremental_o1!` used the wrong stride

`gradient_workspace.jl`'s `dest_contrib_incremental_o1!` (the in-place primitive
`composite_gradient_at_fast_pooled` depends on, via `lfix_incremental_at_ws!`) computed
`d1w = d + (wo - 1) * D` where `D = cache.D` is the **origin** count. Every other site building
this exact linear index into `λstar`/`γ.P` — `build_lfix_base_cache`'s own `CONST_d`/`contrib0`,
and the non-mutating `dest_contrib_incremental_o1` in `lfix_incremental.jl` — uses
`cache.Ddest`, the **destination** count, per `build_lfix_base_cache`'s own documented
column-major convention. `D == Ddest` for every square context (D=4 always; D=20 only under the
legacy `destination_sample=:all_legacy` opt-out), which silently masked this for as long as square
contexts were the only ones exercised.

It is **wrong** under the CURRENT real-D20 production default
(`destination_sample=:exclude_row`, D=20 origins / Ddest=19 destinations). Confirmed live: at a
real D=20/W=80,000 calibration point, the buggy formula produced per-draw contribution values
differing from the correct (non-mutating) function by up to ~98 (correct values are of order
0.01–0.05), corrupting `composite_gradient_at_fast_pooled`'s entire returned A-block gradient by
up to ~75× in magnitude — not a rounding-level discrepancy, a completely wrong answer with the
wrong sign pattern in several components. See
`docs/key_results_shared_a_gradient_2026-07-27/d20_debug_isolation_dest_contrib_o1_bug_repro.log`
for the exact repro (`dest_contrib_incremental_o1` vs `dest_contrib_incremental_o1!` at the same
`(o,d,θ_full)` diverging by 97.6 and 78.7 in max-abs-diff at the very first probed coordinate).

Fixed with a one-line change (`gradient_workspace.jl`, commit `281e830`): `cache.Ddest` instead of
`D`. D=4 regression (`D == Ddest` there, so the fix is a byte-for-byte no-op): confirmed unchanged
via `test_shared_a_gradient.jl`. D=20: confirmed the fix makes `economic_A_gradient!` (which shares
this corrected code path) bit-identical to the trusted `composite_gradient_at_fast` reference.

**This was never caught before because `composite_gradient_at_fast_pooled` was only ever wired as
an OPT-IN alternative for the unrestricted family, never a hard default** (per the prior
allocation audit's own finding) — real exposure at D=20/`:exclude_row` scale was apparently never
exercised end-to-end with a correctness check against the reference gradient. This is exactly the
kind of "assumed two things are close without diffing them" failure mode this project's own
memory (`feedback-gravity-elimination-zero-is-not-calibration`) warns about in a different context
— found here by actually running the D=20 correctness gate this task's own scope required, not by
inspection.

### Bug 2 (FOUND, NOT FIXED — disclosed, out of scope): `composite_gradient_at_fast_buffered`/`_pooled` hard-crash under the current D=20 default

Independently of Bug 1: both `composite_gradient_at_fast_buffered` (`lfix_buffer_reuse.jl`) and
`composite_gradient_at_fast_pooled` (`gradient_workspace.jl`) contain a hardcoded
`z0 = log.(reshape(x_free0[2:end], D, D))` (square reshape) where the correct, Ddest-aware form —
used by `composite_gradient_at_fast` itself — is `reshape(x_free0[2:end], D, Ddest)`. At the real
D=20/`:exclude_row` production default (D=20, Ddest=19, so `x_free0[2:end]` has length 380, not
400), this **throws `DimensionMismatch: new dimensions (20, 20) must be consistent with array
length 380`** — a loud crash, not a silent wrong answer, but a crash nonetheless. Confirmed live in
`docs/key_results_shared_a_gradient_2026-07-27/d20_final_correctness_and_allocation_comparison.log`.

**Implication:** the prior allocation audit's own cited "3690.0 MB/call buffered, 756.5 MB/call
pooled at real D=20/W=80,000" figures (`c10_d20_production_driver.jl:103`'s comment) **cannot have
been measured under the current `:exclude_row` production default** — that call would have thrown
this exact exception. Those numbers must date from before the Part A (2026-07-23) `:exclude_row`
default was introduced, or were measured under an explicit `:all_legacy` override. This is not
this task's bug to fix (out of scope — `composite_gradient_at_fast_buffered`/`_pooled` are
pre-existing, separately-owned files this task's fix-policy says not to touch beyond what's needed)
but it is squarely relevant to this task's own §9 requirement to compare unbuffered/buffered/
pooled/shared at a real D=20 point: **that 4-way comparison is not possible today** because 2 of
the 4 candidates do not run. The comparison actually reported (verdict block above) is
unbuffered-vs-shared, the only pair that both execute correctly at the real production default.

## §2: the D=4 allocation reconciliation (before any new code)

At D=4/W=8000, `composite_gradient_at_fast_pooled`'s cold-bandwidth-cache total (19.12 MB)
decomposes, by direct isolated `@allocated` measurement of each sub-piece, into:

```
build_lfix_base_cache (fresh, once/call):        4.48 MB  (23%, NECESSARY_OUTPUT per the prior audit)
sum(select_bandwidth), all 15 coordinates:       12.64 MB (66%, THE dominant site -- Dict + allocating price cell)
sum(a_block_fd_component_ws!), all 15 coords:     1.99 MB (10%, entirely from 3 same-destination-2-origin coordinates)
RECONCILED SUM:                                  19.12 MB   <-- matches the measured total to <0.01 MB
```

Full detail, including the per-coordinate breakdown that isolated the 1.99 MB to exactly
coordinates k=14/15/16 (the ones sharing destination `d==baseIndex` with the pivot cell), is in
`A_GRADIENT_D20_ALLOCATION_RECONCILIATION_2026-07-27.md` and
`key_results_shared_a_gradient_2026-07-27/d4_allocation_reconciliation.log`.

## Files changed/added

- `full_aod_diag/d4_exact/gradient_workspace.jl` — Bug 1 fix (1 function, `dest_contrib_incremental_o1!`)
- `full_aod_diag/d4_exact/shared_a_gradient.jl` — NEW: `TwoOriginScratch`/`TwoOriginScratchPool`,
  mutating `count_winner_flips!`/`count_winner_flips_multi_top3!`/`select_bandwidth!`,
  `dest_contrib_incremental_top3!`/`lfix_incremental_at_ws2!`/`a_block_fd_component_ws2!`,
  `EconomicAGradientWorkspace`, `economic_A_gradient!`/`economic_A_gradient`
- `full_aod_diag/d4_exact/cm_originzc_production.jl` — ZC-only wiring (`gradient_backend=` kwarg,
  `get_or_build_econ_a_grad_ws`), plus its own p_weights fix
- `full_aod_diag/d4_exact/{cm_production_bundle,cm_meanzc_production,cm_frechet_cplus,fast_range_screen}.jl` —
  p_weights non-allocating diagnostic fix (4 files; origin-ZC's is in the file above)
- `full_aod_diag/d4_exact/test_shared_a_gradient.jl`, `test_shared_a_gradient_d20.jl`,
  `test_originzc_shared_a_gradient_gate.jl`, `bench_shared_a_gradient_reconciliation_d4.jl` — NEW tests/benchmarks

## Candor

Every number in this document set is either directly measured this session (with the log file
cited) or explicitly labeled as not attempted. Where the task's own target ("80% additional
reduction from pooled") could not be met in the literal sense (pooled cannot run at the current
D=20 default to compare against), the comparison actually available (vs unbuffered, the only other
function that runs) is reported instead, with the reason stated plainly rather than silently
substituted.
