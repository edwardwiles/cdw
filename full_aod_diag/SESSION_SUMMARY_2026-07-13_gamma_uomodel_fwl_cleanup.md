# Session summary — UoModel removal, universal γ≡1, FWL gravity simplification, μ-value caching (2026-07-13)

For a future Claude picking this up cold. Unlike [[gamma-d-normalization-and-direct-gp]]'s
2026-07-12 session (purely additive, no production file touched), **this session DOES modify
shared production code**: `moments/hFunction.jl`, `moments/moments!.jl`,
`full_aod_diag/moments_gammanorm.jl`, and several `prepare_cc/`/`lfd/`/`prestep/` files. Read
this before assuming any of those files still work the way older memory notes describe.

**Where the work lives**: branch `feature/sequential-inversion-perf`, two commits on top of
`e3770c4` (fast-forwarded, no merge commit): `ea35b01` (UoModel removal, universal γ≡1, FWL
simplification, vcat cleanup) and `9a4793f` (μ-value caching). Pushed to `origin` as a new branch
(first-ever push of this branch — only `origin/main` existed before). The work was done and
validated on a throwaway branch (`cleanup/uomodel-and-gravity`, off worktree
`trade_robustness_modular_cleanup`) before being fast-forwarded in; that worktree is unused now.

## 1. UoModel=0 removed everywhere

`UoModel` (a toggle for a `U_od`-indexed, D²-wide draws variant vs the `U_o`-indexed, D-wide
`UoModel=1` variant) has never been used with `UoModel=0` in practice — every driver script in the
repo hardcodes `UoModel=1`. Removed the parameter and every `UoModel==0`/`UoModel==1` branch from
`moments/`, `prepare_cc/`, `prestep/master_prestep.jl`, `cc_algo/ccOuter.jl`, and `lfd/`.

**Deliberately NOT touched**: the analytic-Jacobian path (`moments/moments_Jacobian!.jl`,
`moments/hFunction_jacobian.jl`, the `_Jacobian!` variants inside
`GravityMomentFirstApproach!.jl`/`pairewiseIndependenceMoment!.jl`) — this is legacy/dead
(`use_Jacobian=0` in every driver, superseded by ForwardDiff per
[[derivative-algo-experiments]]) and its indexing is intricate enough that editing it without
dedicated test coverage wasn't worth the risk for a cleanup task.

**Also NOT touched**: `full_aod_diag/ad_benchmark/*.jl` and
`full_aod_diag/outer_cache_forwarddiff/minimal_enzyme_nan*.jl` — historical one-off diagnostic
scripts (concluded Enzyme/type-stability experiments) that call `hFunction!`/`hFunctionCounter!`
directly with the OLD (pre-this-session) argument list. **These will now error (wrong arg count)
if run.** Not part of the live pipeline, so left alone rather than touching several more
interlocking one-off files for no production benefit — but if a future Claude goes looking for
one of these and it breaks, this is why.

## 2. Baseline γ normalized to 1 UNIVERSALLY, not just in the gammanorm variant

The user's instruction: "I intend to make gamma = 1 the normalization in all cases going forward."
This is a bigger change than it sounds, because `hFunction!`/`hFunctionCounter!` are SHARED
between `moments/moments!.jl::EK_moments_simple!` (the original A[1,d]=1-normalized path, real
non-unit γ) and `full_aod_diag/moments_gammanorm.jl` (γ≡1). Naively hardcoding γ≡1 into the shared
function would have silently broken `EK_moments_simple!`'s real-γ behavior — the user confirmed
this tradeoff is fine ("I don't understand the question... I intend to make gamma=1 the
normalization in all cases going forward") and `moments/moments!.jl` was updated to match (its own
baseline-γ derivation block — `ΔγA_d`, `Δγμ_d`, `γ[d]=γ_θ[d]*ΔγA_d*Δγμ_d` — is now deleted; γ_prime
i.e. the COUNTERFACTUAL γ' is untouched, still real/differentiated, per the user's explicit
"for counterfactuals we need gamma', but that's different").

**Important type-safety subtlety** (this directly touches the "genuinely different real hazard"
flagged in [[outer-cache-forwarddiff-audit]] about `UPow_scratch` reuse gated on
`eltype(γ)===Float64`): `hFunction!`'s internal scratch arrays (`pricesTemp`, `constCons`, ...)
used to be typed via `eltype(γ)`, which happened to always equal `eltype(θ)` by construction
(γ was `copy(γ_θ)`-derived). Once γ is deleted from `hFunction!`'s signature, that type signal is
gone — and it was NOT safe to just drop it, because those scratch arrays must hold whatever type
`AodPow` produces (Dual, when `Aod_θ` is under ForwardDiff). Fix: `hFunction!` now computes
`T = promote_type(eltype(Aod), eltype(UPow), eltype(Uσ))` internally and uses that instead —
correct regardless of which of Aod/UPow/Uσ (if any) is Dual-typed, with no dependency on a γ
array's type at all. `hFunctionCounter!`'s OWN "γ" parameter is actually γ_prime and was left
completely alone (still real, still needed).

**Validated**: exact-match numerical diff of `hFunction!`/`hFunctionCounter!`/`newGravityMoment!`
against the pre-edit code on identical random D=4 inputs (max abs diff ~2.8e-16 — pure roundoff,
not approximate agreement) — see `scratch/orig_run.jl`/`new_run.jl` pattern in that session's
scratchpad (not committed, reproducible from `git show e3770c4:<path>` if needed again). Also ran
`full_aod_diag/check_gammanorm.jl` end-to-end (no KNITRO needed) — both `EK_moments!` (general
path) and `EK_moments_gammanorm!` produce finite output, `ForwardDiff.jacobian` succeeds cleanly.

## 3. FWL simplification of the gravity moment — verified, applied in two places

Claim checked (user's, from first principles): `newGravityMoment!`'s `Wτ = withinTransform(τ)` is
already orthogonal to row/column means by construction, so `Σ Wτ·within(x) = Σ Wτ·x` for ANY x —
meaning within-transforming the A-side (`WA = withinTransform(Aod)`) is provably redundant; dot
`Wτ` straight into raw `log.(Aod)` instead. **Verified numerically to ~1e-15 across D=3..10**
(generic random matrices, and again with the actual `AodPow=A^(-μ)` functional form) — this is an
UNCONDITIONAL identity, not specific to μ-fixed runs. Applied in `moments/newGravityMoment!.jl`
(now takes `Aod` and no longer needs `U`/`γ`/`UoModel` args either, all now genuinely unused once
the UoModel==0 branch was deleted) and in
`full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl`'s `make_gravity_grad` (the
`ForwardDiff.gradient`-based gravity gradient used by the "REF"/full-theta-AD path — see §5 below;
this is a simplification of the function BEING differentiated, `make_gravity_grad` still calls
`ForwardDiff.gradient` exactly as before, just with less work per call and a shorter chain rule
through `Aod_θ`).

**A real caveat surfaced while double-checking a related, stronger simplification the user
proposed** ("can we skip forming AodPow's `^(-μ)` entirely and use `ln(cHat)-ln(Aod_θ)` directly,
since μ is just an overall positive scale on the whole moment"): this collapse to
"μ cancels" is EXACT when `θConstant==1` (μ truly fixed, `Aod≡Aod_θ`, no reparameterization), but
NOT when μ genuinely varies — `moments!.jl`'s Aod-reparameterization formula (used when
`θConstant≠1`, to keep `Aod_θ≡1` matching Frechet at any μ) introduces a μ-INDEPENDENT additive
constant term into `ln(AodPow)` (from a `(...)^(1/μ)` term whose μ-dependence exactly cancels
against the outer `^(-μ)`, leaving a fixed exponent of -1). Verified numerically: this additive
term is elementwise material (max~1.15 in a test case) and does NOT wash out in the
`Σ Wτ·(...)` dot product (came out to -0.55, not ~0). So the "skip AodPow, use Aod_θ/cHat
directly" idea is only exactly valid in the μ-fixed regime — which is what the user actually runs,
but don't apply it blindly if μ ever becomes a free outer parameter.

## 4. μ-value caching for U^(-μ)/Uσ^(-μ)

New: `obj.γ.μPow_cache::Ref{Float64}` (alongside the pre-existing `UPow_scratch`/`UσPow_scratch`
buffers, added in `prepare_cc/buildObjectsForMoments.jl`) plus a new shared helper
`ensure_UPow!` (defined once in `moments/moments!.jl`, used by both `EK_moments_simple!` and
`full_aod_diag/moments_gammanorm.jl`'s two functions). Recomputes `U^(-μ)`/`Uσ^(-μ)` into the
Float64 scratch buffers ONLY if μ's actual value changed since the last call — previously, the
`eltype(θ)===Float64` check (formerly `eltype(γ)===Float64`) only ever controlled whether a
buffer was REUSED FOR ALLOCATION, not whether the power computation was SKIPPED — it ran
unconditionally, every single call, in both the Float64 and Dual paths. There was no
recompute-skipping mechanism for μ at all before this.

**Why value-based, not a plain `eltype(μ)===Float64` type check** (this was the original plan,
revised after inspecting `cc_algo/free_param_map.jl::reconstruct_full`): under `FreeParamMap`
(used by the "new"/Method-B production drivers), a "fixed" μ still comes out **Dual-typed with
all-zero partials** — `reconstruct_full` does `theta_full[i] = T(fixed_vals[k])` where `T` is
whatever type the FREE coordinates forced the whole vector into. A bare `eltype(μ)===Float64`
check would be `false` in exactly the common, intended-to-be-fast case (only `Aod_θ` differentiated,
μ held fixed) — missing the optimization entirely. `ensure_UPow!` instead checks
`μ isa ForwardDiff.Dual && !iszero(ForwardDiff.partials(μ))` — "does μ carry any NONZERO
derivative information" — and only falls back to a fresh per-call Dual computation in that
(genuinely-differentiating-μ) case.

**Validated** (`scratch/check_mu_cache.jl` pattern, reproducible from the driver setup in
`full_aod_diag/check_gammanorm.jl`): (1) cache-cold vs. cache-warm `ForwardDiff.gradient` w.r.t.
`Aod_θ` (μ unchanged) bit-identical, diff=0.0; (2) `ForwardDiff.gradient` w.r.t. μ itself matches
central finite differences to ~2e-12 relative error (μ's own derivative is NOT silently dropped);
(3) changing μ correctly invalidates and recomputes, bit-identical to a cold reference.

## 5. D=4 "why does the analytic-gravity-gradient path need ~4x the iterations" investigation

**Context, and an important clarification of scope**: `full_aod_diag/gravity_tariff.jl` (closed-
form analytic gravity gradient) and `full_aod_diag/run_fullA_D4_production.jl` (the "new"
cached/FreeParamMap/analytic-gravity path vs. "REF"
`PsiObjectiveBundleImplicitMethodBFullA`/full-theta-ForwardDiff path comparison driver) **predate
this session entirely** — added in commit `6ed76c9`, ten commits before this session's branch
point. This session did NOT introduce, choose, or modify the analytic-gradient method; it was
found while validating §3 above and reused as an existing, already-built test harness (verified via
`git diff e3770c4 9a4793f --stat` on `gravity_tariff.jl`/`outer_loop_cached.jl`/
`free_param_map.jl`/`outer_eval_cache.jl`/`run_fullA_D4_production.jl` — empty diff, none touched).

Running that pre-existing driver at `maxit=25`: "new" hits `status=-410, feas_err=7.5e9` (NOT
converged) vs "REF" `status=-400, feas_err=2.1e-8` (converged, within iteration budget). **Ran the
identical script against the unmodified pre-edit code** (`e3770c4`, throwaway worktree) to rule out
a regression from this session's changes — produced the exact same κ (10 significant figures) and
the exact same non-convergence. So this is a pre-existing property of that specific comparison, not
something introduced today.

**Hessian-callback hypothesis: proposed, checked, WRONG.** Initially assumed "REF"'s
`PsiObjectiveBundleImplicitMethodBFullA.hessian!` (an explicit Gauss-Newton-style Hessian via BLAS
`gemm`) explains REF's faster convergence vs. "new" having no such callback. **Checked directly**:
neither `cc_algo/outer_loop_cached.jl` (new) nor `cc_algo/outer_loop_functions.jl::outer_loop`
(REF) calls `KN_set_cb_hess` for the OUTER solve — that KNITRO callback registration only exists
for the INNER loop (`cc_algo/inner_loop_functions.jl`). Both `.opt` files use `hessopt 4`, which
(per [[full-aod-conditioning]]'s prior finding) silently falls back to L-BFGS with NO exact
Hessian callback registered, in BOTH paths. `hessian!` in `PsiObjectiveBundleImplicitMethodBFullA`
appears to be dead code for this driver (not invoked by KNITRO's outer solve) — don't reuse this
hypothesis without re-verifying it's actually wired up somewhere else.

**Gravity gradient correctness: verified, not the cause.** Directly compared the closed-form
analytic gradient (`gravity_tariff.jl::gravity_grad_free!`) against `ForwardDiff.gradient` of the
same FWL-simplified forward formula at a random D=4 test point — **agree to ~3e-16 relative
error**. Rules out "wrong/imprecise gravity gradient" as an explanation.

**Remaining, NOT yet isolated**: the two paths differ in (a) problem dimensionality as KNITRO
sees it — "new" via `FreeParamMap` hands KNITRO a reduced 17-variable problem (1+D², μ/σ/γ_θ
excluded entirely) vs. "REF"'s full 23-variable vector with those same coordinates pinned via
equal bounds instead of removed; (b) "new" uses `OuterEvalCache` (exact-point reuse), "REF" has
none; (c) the gravity constraint's absolute scale differs (raw sumGrav in REF vs. `-sumGrav/N_obs`
in "new" — each internally consistent between its own value and gradient, but different in
absolute magnitude from each other). Any of these can legitimately send an L-BFGS-based solver
down a different-length iterate trajectory even with every individual gradient exact — this is a
known general property of quasi-Newton methods, not necessarily a bug. **Confirmed the two paths
converge to essentially the same point given enough iterations**: reran "new" at `maxit=100`
(instead of 25) — converges cleanly (`feas_err=5.2e-18`, `opt_err=0.0052`,
`κ=0.1628, γ'_focal=0.8989`), closely matching REF's `maxit=25` result
(`κ=0.1636, γ'_focal=0.8984`). So the 4x-iteration gap is real but does not indicate incorrectness;
which of (a)/(b)/(c) actually drives it is unresolved — would need targeted ablations (e.g. REF
with `FreeParamMap`-reduced dimensionality, or "new" with the cache disabled) to isolate.
