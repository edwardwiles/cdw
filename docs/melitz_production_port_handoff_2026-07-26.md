# Handoff: porting the matrix-free Melitz inner solve into permanent production code

Written 2026-07-26, at the request of the user, who asked to fork the CC inner-solve
machinery into permanent, Melitz-owned, `cc_algo`-independent production code (not a
validation-only side artifact), then offered to hand this specific task to a fresh Claude
session given the size of the remaining work and this session's context budget. This
document is that handoff: what exists, why, exactly what remains, and a concrete plan.

## 0. Where the code actually is, right now

Branch `melitz/fullD-delta-star`, repo `trade_robustness_modular`
(`/bbkinghome/edav/gravity_robustness/trade_robustness_modular`). **Nothing from this
session or the prior sorted-tail session has been committed** -- `git status` on this
branch currently shows:

```
 M src/melitz/include_melitz.jl
 M test/melitz/runtests.jl
?? docs/melitz_matrix_free_inner_operator_2026-07-26.md
?? docs/melitz_production_port_handoff_2026-07-26.md   (this file)
?? src/melitz/matrix_free_dual_solve.jl
?? src/melitz/moment_operator.jl
```

`HEAD` is `1078398f0225e6afaf3031bdb9a85d76ce892da1` ("Melitz: parallel crossing-slice
gradient + production wiring + Phase 8 prototype"), which itself is 15 commits ahead of
`cdw/melitz/fullD-delta-star` (unpushed). **The user has NOT asked for a commit or push at
any point in this session** -- per this repo's own standing instruction ("only commit when
explicitly asked"), nothing has been committed. A future session should confirm with the
user before committing/pushing, but should be aware the CURRENT diff (everything below) is
sitting only in this working tree.

Everything described below was validated live and pushed to Dropbox at
`Gravity robustness/Analysis/Server Output/melitz_matrix_free_operator_session_2026-07-26`
(docs, provenance, and runnable key-result scripts with their actual output logs).

## 1. What already exists and is validated (this session, 3 parts)

### 1.1 `src/melitz/moment_operator.jl` -- the matrix-free moment operator

`MelitzMomentOperator`: precomputes, once per outer point (`melitz_update_moment_operator!`),
a per-draw bin index (`bin[s,o] = #{d : cutoff_od < z_so}`, built via one `O(W+D)` merge
sweep, no per-draw search), destination cutoff order/coefficients, and the dense focal-link
column. On top of that:

- `mul_G!(u, op, zeta, mu)` -- matrix-free `u = -zeta - dot(G[w,:],mu)` (the fixed-dual
  scalar argument every inner objective/constraint eval needs).
- `mul_Gt!(g, op, v)` -- matrix-free `g = G'*v` (the moment-gradient contraction every inner
  gradient eval needs).
- `melitz_same_origin_weighted_block!` / `melitz_cross_origin_weighted_block!` /
  `melitz_full_weighted_gram!` / `melitz_full_weighted_gram_parallel!` -- the COMPLETE
  matrix-free Hessian `G_full'*Diagonal(S)*G_full` (`G_full=[ones(W) G]`), one triangle
  only, upper-triangle-fill discipline chosen because that's exactly what KNITRO's own
  packed-Hessian convention needs (confirmed against `cc_algo/PsiObjectiveBundle.jl`'s own
  `hessian!`).

All zero-allocation after construction (confirmed via `@allocated`, both a first zero-alloc
proof and a subsequent GENUINE bug found+fixed: a naive `Threads.@threads` loop showed
multi-MB allocation despite fully preallocated scratch, traced to Julia closure-boxing of
variables merely LIVE around the parallel region -- fixed by extracting BOTH the
per-iteration body AND the surrounding serial prep/finish code into standalone top-level
functions; final residual is a fixed, `W`/`D`-independent `~13KB` per call, confirmed
identical at D=4/D=10/real-D20 -- pure `Threads.@threads` task-spawn overhead, not a
data-scaling violation).

Validated against `melitz_moments!`'s own dense output to machine precision at D=4/D=10/real
D=20/W=80,000. Real-D20 timing: `mul_G!` `~1.9ms`, `mul_Gt!` `~2.8ms`, full Hessian
`~28ms` serial / `~12ms` parallel (16 threads) -- vs `~22-480ms` for the equivalent dense
`BLAS.gemv!`/`gemm!` arithmetic GIVEN A PREBUILT `G`, and `~1.47-1.49s` just to build `G`
in the first place.

### 1.2 `src/melitz/matrix_free_dual_solve.jl` -- freestanding real-KNITRO validation

`MelitzMatrixFreeDualBundle` + `build_melitz_matrix_free_dual_bundle` +
`melitz_matrix_free_inner_solve` + `melitz_matrix_free_moment_residuals`. A **freestanding**
struct/functor/KNITRO-driver that mirrors `cc_algo/PsiObjectiveBundleDelta`'s functor and
`inner_loop_KNITRO`'s own KNITRO API sequence line-for-line, using ONLY `mul_G!`/`mul_Gt!`/
`melitz_full_weighted_gram!` -- but it does NOT touch `cc_algo` at all (no subtyping, no
extended generic functions, no edits). Deliberately included ONLY inside the
`KNITRO_AVAILABLE`-gated block (both in the test suite and any script), not in
`include_melitz.jl`'s unconditional list, because it does `using KNITRO` at file scope.

**Validated by actually driving a real KNITRO solve to convergence** and comparing against
`build_melitz_psi_bundle`/`melitz_recover_lfd` (the existing dense production path) on the
IDENTICAL `(p,eq,cf,z_draws)` fixture:

| | D=4, W=3000 (or 20,000 in the formal test) | real D=20, W=80,000 |
|---|---:|---:|
| `nStatus` | `0` both | `0` both |
| raw objective abs diff | `7.9e-17` | `2.27e-16` |
| dual `x` max abs diff | `2.35e-13` | `4.06e-10` |
| LFD weights max abs diff | `1.92e-16` | `4.20e-17` |
| **wall-clock speedup (complete real KNITRO solve)** | **`~13x`** | **`19.2x`** |

**One resolved subtlety, not a bug**: `PsiObjectiveBundleDelta.find_smallest` defaults
`true`; `inner_loop`'s own WRAPPER (not the functor) negates the raw KNITRO objective when
set. `melitz_matrix_free_inner_solve` returns the pre-wrapper raw value directly. The first
comparison run showed opposite-sign, equal-magnitude "Delta" values -- resolved by
comparing against `-lfd_dense.Delta`, which then matches to machine precision. Documented in
both the test and the report so this isn't mistaken for a bug again.

Full derivation, every number, and the complete honest "what this does NOT establish" list:
`docs/melitz_matrix_free_inner_operator_2026-07-26.md` (Sections A-J). **Read that
document's Section I in full before starting the production port** -- it has the exact
scope boundary of what's validated (fixed-theta inner dual solve only; no outer
theta-gradient branch; no complementarity constraints; single-seed/W).

### 1.3 Test coverage (all currently green, zero regressions)

`test/melitz/runtests.jl`: 48 top-level testsets, all passing, including "Matrix-free
moment operator (2026-07-26)" (12,445 assertions: objective/gradient/Hessian, serial and
parallel, D=4/D=10/real-D20) and a new KNITRO-gated testset validating
`MelitzMatrixFreeDualBundle` through a real solve at D=4 (folded into the "Pareto
data-only calibration" outer testset's own count, which is why it doesn't show as its own
named summary line -- Test.jl aggregates nested testset counts into the parent's single
row; this is normal, not a sign anything is missing).

## 2. Existing PRE-session production wiring (2026-07-25, already in the codebase, NOT
## built this session -- important context)

The prior day's sorted-tail session already added OPT-IN (not default) production kwargs:

- `moment_backend::Symbol=:dense_reference` on `build_melitz_psi_bundle`/
  `build_melitz_psi_bundle_from_calibration`/`melitz_calibration_outer_ctx` -- passing
  `:sorted_tail_serial`/`:sorted_tail_parallel` builds a `MelitzSortedTailContext` once and
  dispatches `melitz_moments_adapter!` through the sorted-tail fast path instead of the
  dense per-draw `melitz_firm` loop (`~11-19x` moment-construction speedup, machine
  precision, validated end-to-end through a real KNITRO inner solve already).
- `gradient_backend::Symbol` on `solve_melitz_finite_delta_bound`/`melitz_fixed_point_probe`/
  `build_melitz_implicit_bundle` -- `:B_direct_argument_sorted_serial`/`_parallel` (in
  `src/melitz/sorted_crossing_gradient.jl`) use the SAME sorted-tail context to skip
  provably-inactive draws when constructing the outer theta-gradient's touched-column
  probes (`~3-28x` outer-gradient speedup depending on threading, validated exact).

**These are NOT yet the DEFAULT** -- every existing script must opt in explicitly. Given
the user's stated goal ("I want the Melitz production code to be using the new way"), a
prerequisite/parallel task to the Hessian/inner-solve port below is simply: **flip these
two defaults** (`moment_backend` default -> `:sorted_tail_serial` or `_parallel`;
`gradient_backend` default -> the sorted variant) in `build_melitz_psi_bundle`/
`build_melitz_psi_bundle_from_calibration`/`build_melitz_implicit_bundle`/
`solve_melitz_finite_delta_bound`. This is LOW-RISK and INDEPENDENT of the much larger
inner-solve port (Section 3) -- it could be done first, quickly, as a standalone win,
subject to running the full test suite (which already exercises both backends at every
tested scale) to confirm the flip changes no reported economics, only speed.

## 3. What the user is now asking for (this session's final request)

Verbatim: *"I'd like to get this into production code for the Melitz model. I think we
should have essentially no shared functions or code between the Ricardian model and the
Melitz model. So I'd suggest you basically not worry about testing for Ricardo regressions,
and just create a separate version of the cc_algo function for Melitz. It should have the
same settings (at least for now)... but should have these customized aspects for Melitz
like the Hessian callbacks etc. I want to ensure these gains don't get lost across
branches... I want the Melitz production code to be using the new way of building moments,
doing the outer gradient, and doing the inner KNITRO solves."*

Read literally, this is THREE things, of which two (moments, outer gradient) already exist
as opt-in code (Section 2) and just need their DEFAULTS flipped plus, per this request, a
now-explicit mandate to make them **Melitz-owned, not shared with `cc_algo`** wherever they
still route through it. The third (inner KNITRO solves) does not exist as production code
at all yet -- Section 1.2's bundle is a validation-only artifact. The user has explicitly
said Ricardian-model regression testing is NOT required -- only Melitz's own (already
extensive) test suite needs to stay green.

## 4. Survey of what actually depends on `cc_algo` in Melitz production code (done this
## session, so the next session does not have to re-derive it)

**The single most useful fact found this session**: grep every `src/melitz/*.jl` function
signature (not docstrings/comments) for an explicit `obj::PsiObjectiveBundle*` type
annotation --

```
grep -rn "::PsiObjectiveBundle" src/melitz/*.jl
```

returns **zero function signatures** -- every hit is inside a docstring or comment. **Every
Melitz function that takes a bundle argument (`obj`) is written duck-typed, with no type
annotation at all.** This means: if a new Melitz-owned bundle type exposes the SAME field
names and the same callable-functor signature `obj(x, g=Float64[]; h=Float64[])` that
`PsiObjectiveBundleDelta`/`Implicit` currently expose, EVERY existing Melitz consumer
function should work UNCHANGED, with zero source edits, purely via duck typing. This is the
single biggest de-risking fact for this port -- confirm it still holds (re-run the grep)
before assuming it, since new code may have been added since this check.

**What still needs real design work, because it reads/writes the DENSE `obj.H` directly**
(not just through the functor), found by grepping `obj\.H\b` across `src/melitz/*.jl`:

1. **`direct_gradient.jl`'s `_base_arg0!`** (used by BOTH `:B_direct_argument_serial/
   _parallel` AND, transitively, every sorted-crossing backend in
   `sorted_crossing_gradient.jl`): `BLAS.gemv!('N', 1.0, @view(obj.H[:, 2:1+outer_constr_index]),
   -x, 0.0, arg0_base)`. **Fix is straightforward**: since `_base_arg0!` is NOT type-annotated,
   add a SECOND, more-specific method `_base_arg0!(arg0_base, obj::MelitzCCBundle, x)` that
   calls `mul_G!(arg0_base, obj.op, x[1], @view(x[2:end]))` instead -- Julia's multiple
   dispatch picks the specific method automatically for the new bundle type and leaves the
   generic (dense) method untouched for anything still using the old bundle. Zero risk,
   additive.

2. **`direct_gradient.jl`'s `_direct_coordinate_grad`** (the actual per-coordinate gradient
   probe) reads `obj.H[w, Gbase_col_offset]` in a **dense, per-row loop** to get the BASE
   value of each touched trade/link column, so it can apply only the DISPLACED-minus-BASE
   delta (`u_plus[w] -= lam_k*(Gp[w,idx]-gbase)`) rather than recomputing the full `u` from
   scratch. **This is the one place needing genuinely new code, not just a dispatch swap.**
   For a matrix-free bundle, `gbase` for cell `(o,d)` at row `w` is EXACTLY
   `op.coef[o,d] * op.sorted_ctx.z_power_original[w,o] * (op.bin[w,o] >= op.rank[d,o] ?
   1.0 : 0.0) - op.lambda[o,d]` -- reconstructible from fields `MelitzMomentOperator`
   ALREADY carries, no new state needed. Write a small helper, e.g.
   `_base_moment_column!(gbase_vec::Vector{Float64}, op::MelitzMomentOperator, o::Int, d::Int)`
   (or `_link_column!` for the focal-link case, which can just reuse `op.ell` directly --
   the link column IS `op.ell`, no lambda/active-set structure to reconstruct at all), then
   have `sorted_crossing_gradient.jl`'s own analogous function (it has its own copy of this
   loop, check both files) call it instead of indexing `obj.H` when `obj` is the new type.
   Same "add a specific dispatch, keep the generic dense one" pattern as `_base_arg0!`.

3. **`predictor_corrector.jl`** (`melitz_predictor_corrector_continuation`) and
   **`inner_screening.jl`** (`melitz_classified_inner_solve` and others) directly call
   `CS.select_G_from_H(obj, obj.H)` (`CS = CounterfactualSensitivity`, i.e. explicitly
   qualified cc_algo) and `obj.moments!(@view(obj.H[:,1]), G_now, theta, obj.U, obj)` to
   rebuild the dense moment matrix at a NEW theta directly, bypassing `inner_loop_internal`
   entirely (these are the "evaluate at a candidate point without a full KNITRO re-solve"
   diagnostic/screening helpers). **These are the least-well-understood consumers from this
   session's survey and need their own careful read before deciding how to port them** --
   they may need an equivalent "rebuild the operator's fixed-outer-point state at a new
   theta into a scratch operator, without disturbing the bundle's own live `op`" pattern
   (`melitz_update_moment_operator!` on a SEPARATE scratch `MelitzMomentOperator`, then read
   off individual values via `mul_G!`/column-reconstruction as needed) -- NOT attempted or
   designed this session; flagged explicitly as unfinished analysis, not a solved problem
   with a one-line fix like items 1-2 above.

4. **Anything else touching `obj.H`/`obj.moments!`/`CS.` directly** -- the greps above are a
   good start but were not exhaustively walked function-by-function this session for EVERY
   file in the Section survey list (`gradient_lab.jl`, `localized_gradient.jl`,
   `argument_localized_gradient.jl`, `nuisance_profile.jl`, `inner_solve_config.jl`,
   `bounded_cache.jl`). A fresh session should re-run
   `grep -n "obj\.H\b\|CS\.\|CounterfactualSensitivity\." src/melitz/*.jl` and walk every
   hit before assuming the port is complete -- this document does not claim to have found
   every one.

## 5. Concrete design for the new Melitz-owned bundle (sketch, not implemented)

A single struct, e.g. `MelitzCCBundle` (name TBD), replacing what Melitz currently gets from
BOTH `PsiObjectiveBundleDelta` and `PsiObjectiveBundleImplicit` (Melitz's own usage of the
two is nearly identical -- same functor math; the only difference is `PsiObjectiveBundleImplicit`'s
`inner_loop_internal` returns `H[1,1]*(-1)^find_smallest` -- literally `(gamma_prime_target-1)`,
Melitz's own `K` convention, `moments.jl`'s documented placeholder -- instead of the raw
KNITRO objective, because the OUTER driver built on top of it optimizes over `theta[1]`
subject to a `Delta<=budget` constraint, not `Delta` itself). Suggested fields (matching
`MelitzMatrixFreeDualBundle`'s already-built superset, Section 1.2):

```julia
mutable struct MelitzCCBundle
    op::MelitzMomentOperator
    mode::Symbol                 # :delta or :implicit -- controls inner_loop_internal's return convention
    M::Int; d::Int; outer_constr_index::Int
    U::Matrix{Float64}           # kept for consumers that read obj.U directly (size checks etc.)
    lower_limit::Float64
    use_cached_x::Bool
    find_smallest::Bool
    x::Vector{Float64}
    arg0::Vector{Float64}; arg1::Vector{Float64}; arg2::Vector{Float64}
    Hfull::Matrix{Float64}       # (d+1)x(d+1) Hessian scratch
    H_save::Float64              # only meaningful in :implicit mode
    moments!::Function           # kept for API parity even though it may go unused internally
    γ::Any                       # ctx NamedTuple, unchanged from current usage
    inner_loop_opt::String
end
```

Own, Melitz-only versions of (all currently `cc_algo`-owned, generic across Ricardian +
Melitz -- the user's explicit directive is to NOT share these anymore):

- The functor (`(Q::MelitzCCBundle)(x, g=Float64[], θ=Float64[]; h=Float64[], constr=..., jac=...)`)
  -- port `MelitzMatrixFreeDualBundle`'s already-built, already-validated functor (Section
  1.2) directly; add the `:implicit`-mode `H_save` convention and (only if some LIVE
  production caller actually needs it -- audit first, Section 4 item 4's own open question)
  the `constr` branch (recall PART 3's own finding: for Melitz, `outer_constr_index == d+1`
  always, making the generic `outer_constr_index<d` "extra outer-constraint moments" branch
  VACUOUS -- confirm this still holds before deciding whether `constr`/`jac` need any real
  implementation at all, or can be left as unreachable dead code exactly as they already are
  for the existing dense bundles per PART 1's own audit).
- `melitz_inner_loop_knitro(bundle)` / `melitz_inner_loop(bundle, theta)` -- Melitz-owned
  ports of `inner_loop_KNITRO`/`inner_loop`, built directly from
  `melitz_matrix_free_inner_solve` (Section 1.2) plus the `find_smallest` wrapper convention
  folded IN (rather than left to the caller, unlike the validation harness) so it's a true
  drop-in replacement for `inner_loop`.
- Divergence conjugate `Psi!`/`dPsi!`/`ddPsi!` -- already copied verbatim under Melitz-local
  names in `matrix_free_dual_solve.jl` (`_mf_Psi!` etc.) -- reuse as-is, or rename without
  the underscore-prefix if this becomes permanent production code (a leading underscore
  conventionally signals "private/internal" in this codebase; a production divergence
  function likely wants a public name).
- `guard_enter_inner_solve!`/`guard_exit_inner_solve!` (currently cc_algo's own concurrency
  guard, preventing two inner KNITRO solves running concurrently across `Threads.@threads`)
  -- per the user's "no shared code" directive, Melitz needs its OWN copy of this guard too,
  not a reference into `cc_algo/parallelism_guards.jl`. Small, mechanical port.

Then rewire the actual construction entry points:

- `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration` (`delta_star.jl`/
  `pareto_calibration.jl`) -- construct `MelitzCCBundle` (`mode=:delta`) instead of
  `PsiObjectiveBundleDelta`. `melitz_recover_lfd`/`melitz_recover_lfd_from_solution` need
  their own moment-residual recovery to use `mul_Gt!` instead of a dense `G` (this session's
  `melitz_matrix_free_moment_residuals`, Section 1.2, is already exactly this function --
  port it in).
- `build_melitz_implicit_bundle` (`finite_delta_outer.jl`) -- construct `MelitzCCBundle`
  (`mode=:implicit`) instead of `PsiObjectiveBundleImplicit`.
- Keep the OLD dense-bundle code paths (`PsiObjectiveBundleDelta`/`Implicit` +
  `melitz_moments_adapter!`'s `:dense_reference` backend) fully intact and callable --
  per this session's own PART 1-3 reports' recommendation, the dense path remains valuable
  as a trusted diagnostic/cross-check, and per this repo's own standing practice (never
  delete a working, tested reference implementation without being asked). The NEW bundle
  becomes the DEFAULT these constructors return; a caller wanting the old dense path
  explicitly could still get it via a kwarg (e.g. `backend=:dense` vs `backend=:matrix_free`)
  -- naming/mechanism TBD by the implementing session.

## 6. Recommended order of work for the next session

1. **Re-run the surveys in Section 4** to confirm they still hold (code may have moved).
2. **Flip the two ALREADY-EXISTING defaults** (Section 2: `moment_backend`,
   `gradient_backend`) to their sorted variants. Run the full test suite. This alone
   delivers real, immediate, low-risk production value and can be done/committed
   independently of everything else.
3. **Design and implement `MelitzCCBundle`** (Section 5) as a genuinely new, Melitz-owned
   type -- start from `MelitzMatrixFreeDualBundle` (already built, already validated) and
   extend it with the `:implicit` mode and Melitz's own inner-loop/guard functions.
4. **Add the dispatch-based `_base_arg0!`/base-column-reconstruction methods** (Section 4
   items 1-2) so the EXISTING `direct_gradient.jl`/`sorted_crossing_gradient.jl` outer-
   gradient code works unchanged against the new bundle type (this is the biggest
   leverage-per-line-of-code item in the whole port, per this session's survey).
5. **Investigate and port `predictor_corrector.jl`/`inner_screening.jl`'s direct
   `obj.H`/`CS.select_G_from_H` usage** (Section 4 item 3) -- the least-understood piece;
   budget real analysis time here, don't assume it's a quick dispatch swap like items 1-2.
6. **Rewire `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`/
   `build_melitz_implicit_bundle`** to construct `MelitzCCBundle` by default.
7. **Full regression pass**: the ENTIRE existing Melitz test suite (48 testsets, currently
   all green) must still pass with the new bundle as the default -- this is the actual
   correctness gate the user has authorized in place of Ricardian-model testing. Do not
   consider the port done until every existing Melitz test (not just the new matrix-free
   ones) passes against the new default path.
8. **Re-validate the headline real-D20 KNITRO-solve comparison** (Section 1.2's own table)
   one more time against the FINAL wired-in production entry points (not the standalone
   validation bundle) to confirm the production integration didn't silently reintroduce a
   discrepancy.
9. Update `docs/melitz_matrix_free_inner_operator_2026-07-26.md`'s own Section J
   recommendation (currently: "not recommended as a silent default change") once this work
   lands -- that sentence will need to be reversed once the port is real and tested.
10. Confirm with the user before committing/pushing (nothing in this branch has been
    committed yet, Section 0).

## 7. Things NOT to re-litigate (already settled, cited so the next session doesn't
## re-derive or contradict them)

- The exact algebra (`G_trade = R_trade - ones(W)*lambda'`, the bin/rank/order construction,
  the rank-one Hessian correction) is DERIVED and VALIDATED -- do not re-derive, cite
  `docs/melitz_matrix_free_inner_operator_2026-07-26.md` Sections A-D.
- `melitz_full_weighted_gram!`'s one-triangle-only convention is INTENTIONAL (direct user
  instruction, matches KNITRO's own packed-Hessian format) -- do not "fix" it to fill both
  triangles.
- The `find_smallest` sign convention (Section 1.2) is a REPORTING wrapper, not a
  computation bug -- if a future comparison shows opposite-sign equal-magnitude values,
  check this FIRST before assuming a new bug.
- `CLAUDE.md`'s own standing "A_od==1 is not calibration" warning is UNRELATED to this
  work but remains in force for any other Melitz/gravity work in this repo -- not relevant
  to the CC dual-solve port itself, mentioned only so it isn't confused with this document's
  own "resolved subtlety" note above.
