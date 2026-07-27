# Stage 2 (2026-07-24 outer-benchmark-correction session, main prompt Sections 8-10): the
# profiled/continuation alternative to the constrained finite-delta search --
#
#     Delta_profile(g) = min_eta Delta(g, eta)
#
# where `eta` is some admissible subset of the `theta_free[2:end]` A/f nuisance coordinates
# (main prompt's own `2D^2-2`-length free gravity-pivoted vector, `delta_star.jl`), `g =
# theta_free[1]` is PINNED (box lobnd==upbnd), and the objective is `Delta(theta)` ITSELF --
# genuinely re-solved via the SAME inner CC dual problem `evaluate_melitz_delta`/
# `melitz_recover_lfd` already use (`PsiObjectiveBundleDelta`, `build_melitz_psi_bundle`),
# not a proxy/lower-bound. Reuses:
#
#   - the existing `obj_inner::PsiObjectiveBundleDelta` bundle AS-IS (no new bundle type --
#     `inner_loop(obj_inner, theta)` already returns `Delta(theta)` directly in the correct
#     sign/scale, `melitz_recover_lfd`'s own docstring) -- no `PsiObjectiveBundleImplicit`
#     construction is needed at all for this problem, unlike the finite-delta outer NLP.
#   - `direct_gradient.jl`'s fixed-dual gradient backends (`make_melitz_gradient_delta_direct_serial/_parallel`),
#     called at the JUST-CONVERGED optimal dual `x` -- a genuine envelope-theorem-EXACT
#     gradient of `Delta(theta)` here (unlike the finite-delta outer NLP's own use of the
#     same machinery at a possibly-SUBOPTIMAL certifying dual for an AboveEvaluationCap
#     rejection, finite_delta_outer.jl) since `x` is a true local optimum of the inner
#     problem at every accepted trial point.
#   - `affine_cutoff.jl`'s native `:linear` cutoff-constraint machinery
#     (`build_melitz_affine_cutoff_system`, `KN_add_con_linear_struct`) -- the SAME 400
#     domestic-support/export-selection rows, registered exactly as `finite_delta_outer.jl`'s
#     own `:linear` backend does, with zero per-iterate cost.
#
# No divergence-budget constraint is registered at all here (there is nothing to bound --
# `Delta(theta)` IS the objective being minimized).
#
# 2026-07-26 closure session (governing prompt Phase 6): ported to support the
# production-fast matrix-free `MelitzCCBundle` (`obj_inner` may now be EITHER
# `PsiObjectiveBundleDelta` (legacy dense) or `MelitzCCBundle` (matrix-free, `mode=:delta`)
# -- `build_melitz_psi_bundle`'s own default). Nothing about the optimization algorithm
# changed (still the identical box-constrained direct-gradient KNITRO NLP; no trust-region or
# predictor-corrector logic was introduced, per the governing prompt's own explicit
# instruction not to): only the THREE bundle-specific internals were swapped for the
# bundle-agnostic dispatch pair already established during the 2026-07-26 production-port
# session (`cc_bundle.jl`) -- `inner_loop(obj_inner,theta)` -> `melitz_bundle_inner_loop`
# (new, this session, composed from the existing `melitz_bundle_prepare_at_theta!`/
# `melitz_bundle_inner_solve!` pair), and `obj_inner.H`'s direct read/write in the exact-point
# cache -> `melitz_heavy_snapshot`/`melitz_heavy_restore!` (already existed, already
# bundle-agnostic). `direct_gradient_fn` needed NO changes at all -- it was already
# bundle-agnostic via multiple dispatch on the generic `obj` parameter it already had.

using KNITRO
using LinearAlgebra: norm

"""
    melitz_nuisance_free_mask(ctx; block::Symbol) -> BitVector

Builds the length-`n` (`n = 2D^2-2`, `affine_cutoff.jl`'s `melitz_free_dim`) free-coordinate
mask for a Stage 2 nuisance-minimization run, main prompt Section 8's three matched
problems. `theta_free[1]` (`g`) is ALWAYS pinned (`false`) -- Stage 2 profiles `Delta` at a
FIXED `g`, never optimizes over it (that is Stage 1's job).

  - `:A_only`: free = the `D^2-1` A-block coordinates only (`theta_free[2:1+nA]`).
  - `:f_only`: free = the `D^2-2` f/q-block coordinates only (`theta_free[2+nA:end]`).
  - `:full`: free = every coordinate except `g` (both blocks).
"""
function melitz_nuisance_free_mask(ctx; block::Symbol)
    block in (:A_only, :f_only, :full) || throw(ArgumentError(
        "melitz_nuisance_free_mask: block must be :A_only, :f_only, or :full, got $block"))
    n = melitz_free_dim(ctx)
    nA = ctx.D^2 - 1
    mask = falses(n)
    if block == :A_only
        mask[2:1+nA] .= true
    elseif block == :f_only
        mask[2+nA:end] .= true
    else
        mask[2:end] .= true
    end
    return mask
end

"""
    melitz_build_nuisance_profile_callbacks(obj_inner, ctx; gradient_backend, h, on_eval) -> NamedTuple

Builds the `cb_F!`/`cb_G!` pair for the Stage 2 outer NLP: `cb_F!` runs a genuine (warm-
continued, per `obj_inner`'s own `use_cached_x`/`x` cache) inner CC dual solve at the trial
`theta` (`CounterfactualSensitivity.inner_loop`, unmodified) and sets the KNITRO objective to
`Delta(theta)` directly; `cb_G!` re-solves at the same `theta` (matching this repo's own
`eval_fcga=no` convention, `finite_delta_outer.jl`'s identical FC/GA split) and fills the
objective gradient via the fixed-dual direct backend at the newly-converged optimal dual.

An inner-solve failure (`nStatus` outside the accepted set) throws a `DomainError`, caught by
KNITRO.jl's own `_try_catch_handler` and converted to a proper evaluation-error return code --
the SAME convention `finite_delta_outer.jl` uses for a genuine (uncertified) numerical
failure. There is no `AboveEvaluationCap`-style finite-fallback here: this problem has no
budget to certify a violation against, only a real inner solve succeeding or not.

`on_eval`, if given, is called as `on_eval(theta, Delta, nStatus, kind)` (`kind` is `:fc` or
`:ga`) right before returning -- a zero-risk diagnostic hook, default `nothing`.

`on_start` (2026-07-25 diagnostic addition), if given, is called as
`on_start(theta, call_index, kind)` BEFORE `inner_loop` is invoked -- unlike `on_eval`, this
fires even if the subsequent inner solve never returns (a stall/hang), so a caller
investigating exactly that failure mode has the ATTEMPTED point on record regardless of
outcome. Default `nothing`, zero risk to every existing caller.
2026-07-25 (user-directed follow-up, same session as the initial local-geometry/
continuation work): `exact_cache` ports `finite_delta_outer.jl`'s `MelitzExactPointCache`
mechanism to THIS callback pair -- previously the single biggest confirmed-but-unquantified
inefficiency here (docs/melitz_real_d20_outer_benchmark_2026-07-24.md Section 5.1: "`cb_G!`
always re-solves the inner problem from scratch at every outer iteration instead of reusing
`cb_F!`'s own just-computed dual at the identical `theta`"). The cache mechanism itself is
NOT Implicit-specific -- `melitz_exact_cache_get`/`_insert!` operate generically on any
`obj.H`/`obj.moments!`/`obj.U` triple via `CounterfactualSensitivity.select_G_from_H`, which
is dispatched on the abstract `PsiObjectiveBundle` supertype (`cc_algo/PsiObjectiveBundle.jl:659`)
-- `PsiObjectiveBundleDelta` (this file's own `obj_inner`) satisfies that interface exactly
the same way `PsiObjectiveBundleImplicit` does, so the SAME cache type/functions are reused
verbatim, not reimplemented. Default `nothing` (fresh, call-local, empty cache) mirrors
`melitz_build_finite_delta_callbacks`'s own default exactly.

`on_start`/`on_eval` are extended (additively -- every existing 4-argument call site from
2026-07-25 earlier the same day continues to work unchanged, since Julia dispatches on
argument COUNT and this file has zero call sites yet passing only 4) to also report
`elapsed_s` (wall time of the inner-solve portion ONLY, `0.0` on a cache hit) and
`cache_hit::Bool` -- the direct, measured answer to "how much is the cache actually saving,"
rather than an inferred/guessed contribution.
"""
function melitz_build_nuisance_profile_callbacks(obj_inner, ctx; gradient_backend::Symbol=:B_direct_argument_parallel,
                                                  h::Real=1e-4, on_eval=nothing, on_start=nothing,
                                                  exact_cache::Union{Nothing,MelitzExactPointCache}=nothing,
                                                  forbid_dense_fallback::Bool=false)
    # 2026-07-26 closure session (governing prompt Phase 6): ported to the production-fast
    # matrix-free backend. `direct_gradient_fn` was ALREADY bundle-agnostic before this port
    # (`_base_arg0!`/`_direct_coordinate_grad`/`_direct_coordinate_grad_sorted`, cc_bundle.jl,
    # gained `obj::MelitzCCBundle`-specific methods during the 2026-07-26 production-port
    # session -- `make_melitz_gradient_delta_direct_serial/_parallel`'s closures call them
    # through the generic, untyped `obj` parameter, so multiple dispatch already selects the
    # right method for either bundle type with no changes here). Added the two SORTED
    # backends (governing prompt: "sorted outer/nuisance gradients where applicable") --
    # requires `ctx.sorted_tail_ctx`, exactly the same requirement `finite_delta_outer.jl`'s
    # own `:B_direct_argument_sorted_*` already impose.
    direct_gradient_fn = gradient_backend == :B_direct_argument_serial ? make_melitz_gradient_delta_direct_serial(h) :
                         gradient_backend == :B_direct_argument_parallel ? make_melitz_gradient_delta_direct_parallel(h) :
                         gradient_backend == :B_direct_argument_sorted_serial ? make_melitz_gradient_delta_direct_sorted_serial(h) :
                         gradient_backend == :B_direct_argument_sorted_parallel ? make_melitz_gradient_delta_direct_sorted_parallel(h) :
                         error("melitz_build_nuisance_profile_callbacks: gradient_backend must be " *
                               ":B_direct_argument_serial, :B_direct_argument_parallel, " *
                               ":B_direct_argument_sorted_serial, or :B_direct_argument_sorted_parallel, " *
                               "got $gradient_backend")
    if forbid_dense_fallback && !(obj_inner isa MelitzCCBundle)
        throw(ArgumentError(
            "melitz_build_nuisance_profile_callbacks: forbid_dense_fallback=true (strict " *
            "production-fast mode) but obj_inner is not a MelitzCCBundle (got " *
            "$(typeof(obj_inner))) -- construct obj_inner via build_melitz_psi_bundle(...; " *
            "backend=:matrix_free) (the default) for a strict caller."))
    end
    n_fc_calls = Ref(0)
    n_ga_calls = Ref(0)
    exact_cache = exact_cache === nothing ? MelitzExactPointCache() : exact_cache
    n_exact_cache_hits = Ref(0)
    n_exact_cache_misses = Ref(0)

    # Exact-point cache lookup/insert, mirroring inner_solve_verified_or_fail's own logic
    # in finite_delta_outer.jl (Section 4.2/5.1 there) -- kept local to this file rather than
    # shared, since this callback's accept-set/failure convention (DomainError on ANY
    # rejected nStatus, no AboveEvaluationCap-style fixed sentinel) differs from that file's.
    # 2026-07-26 closure session (Phase 6): `melitz_exact_cache_get`/`_insert!` were ALREADY
    # bundle-agnostic (the "heavy" tier stores whatever `melitz_heavy_snapshot` returns --
    # a dense `Matrix` copy or a `MelitzOperatorSnapshot`, cc_bundle.jl) -- only the two
    # direct `obj_inner.H` reads/writes below were dense-specific; both replaced with the
    # SAME `melitz_heavy_restore!`/`melitz_heavy_snapshot` dispatch `finite_delta_outer.jl`'s
    # own exact-point cache already uses.
    function inner_solve_cached(theta::AbstractVector)
        key = Vector{Float64}(theta)
        hit = melitz_exact_cache_get(exact_cache, key, ctx, obj_inner.U; obj=obj_inner)
        if hit !== nothing
            n_exact_cache_hits[] += 1
            Delta_hit, x_hit, nStatus_hit, H_hit = hit
            melitz_heavy_restore!(obj_inner, H_hit)
            return (Delta_hit, x_hit, nStatus_hit, 0.0, true)
        end
        n_exact_cache_misses[] += 1
        t0 = time_ns()
        val, x, nStatus_raw = melitz_bundle_inner_loop(obj_inner, theta)
        elapsed_s = (time_ns() - t0) / 1e9
        # nStatus comes back as Int32 from the raw KNITRO solve (matching
        # inner_screening.jl's own `Int(nStatus)` conversion at its analogous call site) --
        # melitz_exact_cache_insert!/_get! require Int (Int64) for their `nStatus::Int` field.
        # x is likewise coerced to a concrete Vector{Float64} (matching
        # melitz_classified_inner_solve's own `collect(Float64.(x))`), not assumed to already
        # be exactly that type. Root-caused live: omitting BOTH conversions previously threw a
        # MethodError INSIDE the KNITRO callback on every accepted solve's cache-insert
        # attempt, silently reported by KNITRO as a generic eval error (nStatus=-500) --
        # confirmed via a standalone reproduction, not a test-only artifact.
        nStatus = Int(nStatus_raw)
        x_concrete = collect(Float64.(x))
        if nStatus in (0, -100, -101, -103)
            melitz_exact_cache_insert!(exact_cache, key, Float64(val), x_concrete, nStatus,
                melitz_heavy_snapshot(obj_inner), ctx, obj_inner.U)
        end
        return (val, x_concrete, nStatus, elapsed_s, false)
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        theta = collect(evalRequest.x)
        n_fc_calls[] += 1
        # 2026-07-25 diagnostic addition (additive, zero-risk when nothing): fires BEFORE the
        # (potentially very slow / never-returning) inner_loop call below, so a caller
        # investigating a stall has the ATTEMPTED point on record even if this call never
        # completes -- `on_eval` alone (fires only on return) cannot capture that case.
        on_start !== nothing && on_start(theta, n_fc_calls[], :fc)
        val, x, nStatus, elapsed_s, cache_hit = inner_solve_cached(theta)
        # 2026-07-25 diagnostic addition: on_eval now fires UNCONDITIONALLY (moved ahead of
        # the accept/throw check below), so a caller gets the exact nStatus (and inner_loop's
        # own sentinel val=-1e10) for a FAILED call too, not only a successful one -- needed
        # to distinguish "the KNITRO-native lower_limit bailout fired" from "a routine
        # time/iteration cap was reached" from the nStatus value alone, without which a
        # failure was previously indistinguishable from "never returned at all."
        on_eval !== nothing && on_eval(theta, val, nStatus, :fc, elapsed_s, cache_hit)
        accepted = nStatus in (0, -100, -101, -103)
        accepted || throw(DomainError(theta[1],
            "melitz nuisance-profile FC: inner CC dual solve failed, nStatus=$nStatus -- " *
            "rejecting this trial point (no budget-style finite fallback exists for this problem)"))
        evalResult.obj[1] = val   # Delta(theta) directly -- inner_loop's own sign/scale convention
        return 0
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        theta = collect(evalRequest.x)
        n_ = length(theta)
        n_ga_calls[] += 1
        on_start !== nothing && on_start(theta, n_ga_calls[], :ga)
        val, x, nStatus, elapsed_s, cache_hit = inner_solve_cached(theta)
        on_eval !== nothing && on_eval(theta, val, nStatus, :ga, elapsed_s, cache_hit)
        accepted = nStatus in (0, -100, -101, -103)
        accepted || throw(DomainError(theta[1],
            "melitz nuisance-profile GA: inner CC dual solve failed, nStatus=$nStatus"))
        local_jac = zeros(n_)
        # Genuine envelope-theorem-exact gradient: x is a true local optimum of the inner
        # problem at this theta (unlike finite_delta_outer.jl's AboveEvaluationCap-rejection
        # use of the same backend at a possibly-suboptimal certifying dual).
        direct_gradient_fn(local_jac, theta, ctx, obj_inner, x)
        evalResult.objGrad .= local_jac ./ 1e10   # direct_gradient_fn's own 1e10-scaled convention
        return 0
    end

    return (cb_F! = cb_F!, cb_G! = cb_G!, n_fc_calls = n_fc_calls, n_ga_calls = n_ga_calls,
            exact_cache = exact_cache, n_exact_cache_hits = n_exact_cache_hits,
            n_exact_cache_misses = n_exact_cache_misses)
end

"""
    melitz_register_nuisance_profile_knitro_problem!(kc, ctx, cbset, xIndices, n) -> (cIndices, cutoff_sys)

Registers the 400 affine cutoff rows as TRUE KNITRO linear constraints (`:linear` backend,
`affine_cutoff.jl`, zero per-iterate cost) and the objective-only nonlinear eval callback
(`cbset.cb_F!`/`cb_G!`, `Int32[]` constraint indices -- no eval-callback-registered
constraints at all, matching `cc_algo/inner_loop_functions.jl`'s own pure-objective
registration pattern, `KN_add_eval_callback(kc, true, Int32[], ...)`). `nnzJ=0` is passed
explicitly to `KN_set_cb_grad` (rather than relying on its own nonzero-constraint-count
auto-detection, which would otherwise assume a DENSE Jacobian contribution from this
callback across the model's 400 already-registered NATIVE linear rows -- this callback
contributes zero constraint rows, only the objective gradient).
"""
function melitz_register_nuisance_profile_knitro_problem!(kc, ctx, cbset, xIndices, n::Int)
    cutoff_sys = build_melitz_affine_cutoff_system(ctx)
    m = size(cutoff_sys.C, 1)
    cIndices = KNITRO.KN_add_cons(kc, m)
    KNITRO.KN_set_con_lobnds(kc, m, cIndices, -cutoff_sys.b)
    nnz = m * n
    indexCons_lin = repeat(cIndices, inner=n)
    indexVars_lin = repeat(xIndices, outer=m)
    coefs_lin = vec(permutedims(cutoff_sys.C))
    KNITRO.KN_add_con_linear_struct(kc, nnz, indexCons_lin, indexVars_lin, coefs_lin)

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cbset.cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cbset.cb_G!; nnzJ=0)
    return cIndices, cutoff_sys
end

"""
    MelitzNuisanceProfileResult

Result of `solve_melitz_nuisance_min_delta`. `Delta_min`/`theta_final` are KNITRO's own
terminal point; `r_final` is an INDEPENDENT cold re-verification (`evaluate_melitz_delta`,
`cold=true` -- a fresh KNITRO inner solve from a cleared warm start, never trusting the
outer trajectory's own possibly-warm-contaminated terminal state) -- `r_final.Delta` is the
number that should be reported/compared, not the raw `Delta_min`/`objVal` alone (main prompt
Section 8's own "full verification" requirement).
"""
struct MelitzNuisanceProfileResult
    nStatus::Int
    Delta_min::Float64
    theta_start::Vector{Float64}
    theta_final::Vector{Float64}
    free_mask::BitVector
    wall::Float64
    n_fc_calls::Int
    n_ga_calls::Int
    r_final::MelitzDeltaEvalResult
    n_exact_cache_hits::Int
    n_exact_cache_misses::Int
end

"""
    solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_start; free_mask, radius=0.5,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt,
        outer_loop_opt=<default>, on_eval=nothing, warm_start_x=nothing)
        -> MelitzNuisanceProfileResult

Main prompt Section 8/9: solves `min_eta Delta(g,eta)` at `theta_start[1]` PINNED (box
lobnd==upbnd whenever `free_mask[1]==false`, the ordinary Stage 2 case) and every
`free_mask[k]==true` coordinate free within `theta_start[k] .± radius[k]` -- every OTHER
coordinate is ALSO pinned exactly at `theta_start`'s own value (zero degrees of freedom for
a masked-out coordinate, the SAME "lobnd==upbnd" mechanism `melitz_fixed_point_probe`/
`solve_melitz_finite_delta_bound`'s own vector `theta_box` already use). `radius` may be a
scalar (uniform) or a length-`n` vector.

`warm_start_x`, if given, seeds `obj_inner.x`/`use_cached_x` BEFORE the first callback call
(Section 9's continuation-in-g warm start: the preceding continuation point's own converged
dual). Default `nothing`: whatever `obj_inner`'s cache already holds (an ordinary warm
continuation from the caller's own prior use of `obj_inner`).

`inner_solve_config::MelitzInnerSolveConfig` (2026-07-25 local-geometry/continuation
session, REQUIRED, no default -- see `inner_solve_config.jl`'s file header for the full
incident this closes): this function is THE nested-repeated-inner-solve driver in this
codebase -- every accepted outer trial point triggers at least two full inner CC dual solves
(`cb_F!`/`cb_G!`) against the CALLER-supplied `obj_inner`, whose own construction
(`build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`) may or may not have
set a cap. Confirmed live in a prior session
(`docs/melitz_real_d20_outer_correction_2026-07-24.md` Section 10.4): `obj_inner.lower_limit`
sitting at its uncapped struct default here turned repeated divergent trial points into
~90-second grinds instead of near-instant certified rejections. This function now applies
`inner_solve_config.lower_limit` to `obj_inner.lower_limit` UNCONDITIONALLY at entry --
independent of how `obj_inner` was built -- and restores `obj_inner`'s PRIOR `lower_limit`
in a `finally` block before returning, so a caller sharing `obj_inner` across multiple
purposes (e.g. also using it for an uncapped cold end-of-run reverification elsewhere) is
never surprised by a silently-mutated shared object after this call returns. Pass
`MelitzInnerSolveConfig(:full_value)` explicitly if an uncapped nuisance-minimization run is
genuinely intended -- there is no way to reach the old silent-omission behavior by accident.

`exact_cache::Union{Nothing,MelitzExactPointCache}` (2026-07-25, same-day follow-up,
user-directed): ports `finite_delta_outer.jl`'s exact-point cache to this driver
(`melitz_build_nuisance_profile_callbacks`'s own docstring has the full mechanism) -- `nothing`
(default) allocates a fresh, call-local, empty cache exactly as before this kwarg existed.
`MelitzNuisanceProfileResult.n_exact_cache_hits`/`n_exact_cache_misses` report the MEASURED
effect directly (a `cb_G!` hit at the SAME `theta` a `cb_F!` call just solved is now a
dictionary lookup + one `obj_inner.H` copy, not a second full KNITRO inner solve).
"""
function solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_start::AbstractVector;
                                          free_mask::BitVector,
                                          radius::Union{Real,AbstractVector}=0.5,
                                          gradient_backend::Symbol=:B_direct_argument_parallel,
                                          h::Real=1e-4,
                                          inner_loop_opt::AbstractString,
                                          outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_nuisance_profile.opt"),
                                          on_eval=nothing,
                                          on_start=nothing,
                                          warm_start_x::Union{Nothing,AbstractVector}=nothing,
                                          inner_solve_config::MelitzInnerSolveConfig,
                                          exact_cache::Union{Nothing,MelitzExactPointCache}=nothing,
                                          forbid_dense_fallback::Bool=false)
    n = length(theta_start)
    length(free_mask) == n || throw(ArgumentError(
        "solve_melitz_nuisance_min_delta: free_mask length ($(length(free_mask))) must equal " *
        "length(theta_start) ($n)"))
    theta0 = collect(Float64.(theta_start))

    if warm_start_x !== nothing
        obj_inner.x .= warm_start_x
        obj_inner.use_cached_x = true
    end

    cbset = melitz_build_nuisance_profile_callbacks(obj_inner, ctx;
        gradient_backend=gradient_backend, h=h, on_eval=on_eval, on_start=on_start,
        exact_cache=exact_cache, forbid_dense_fallback=forbid_dense_fallback)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, outer_loop_opt)
    xIndices = KNITRO.KN_add_vars(kc, n)

    r = radius isa Real ? fill(Float64(radius), n) : collect(Float64.(radius))
    lo = copy(theta0)
    hi = copy(theta0)
    @inbounds for k in 1:n
        if free_mask[k]
            lo[k] -= r[k]
            hi[k] += r[k]
        end
    end
    KNITRO.KN_set_var_lobnds_all(kc, lo)
    KNITRO.KN_set_var_upbnds_all(kc, hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, theta0)

    melitz_register_nuisance_profile_knitro_problem!(kc, ctx, cbset, xIndices, n)

    saved_lower_limit = obj_inner.lower_limit
    obj_inner.lower_limit = inner_solve_config.lower_limit
    local nStatus, objVal, theta_final_raw, wall
    try
        t0 = time()
        KNITRO.KN_solve(kc)
        nStatus, objVal, theta_final_raw, _ = KNITRO.KN_get_solution(kc)
        wall = time() - t0
    finally
        KNITRO.KN_free(kc)
        obj_inner.lower_limit = saved_lower_limit
    end

    theta_final = collect(theta_final_raw)
    r_final = evaluate_melitz_delta(theta_final, ctx, obj_inner; cold=true)

    return MelitzNuisanceProfileResult(nStatus, Float64(objVal), theta0, theta_final,
        free_mask, wall, cbset.n_fc_calls[], cbset.n_ga_calls[], r_final,
        cbset.n_exact_cache_hits[], cbset.n_exact_cache_misses[])
end
