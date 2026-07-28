# Melitz-owned permanent production CC bundle + inner KNITRO driver (2026-07-26 production-
# port session). See docs/melitz_production_port_handoff_2026-07-26.md Section 5 for the
# design sketch this file implements, and
# docs/melitz_matrix_free_inner_operator_2026-07-26.md / src/melitz/matrix_free_dual_solve.jl
# for the already-validated matrix-free algebra this file ports into permanent production
# code (same math, copied verbatim -- not re-derived).
#
# NO cc_algo type is subtyped, no cc_algo generic function is extended, no cc_algo file is
# edited. `MelitzCCBundle` is a freestanding struct with its own functor, its own KNITRO
# driver, its own divergence conjugate (`melitz_cc_Psi!`/`_dPsi!`/`_ddPsi!`, copied verbatim
# from `cc_algo/Psi.jl` under Melitz-local names), and its own single-flight concurrency
# guard -- the only infrastructure shared with the Ricardian model is the external
# Julia/KNITRO/BLAS packages, per the user's explicit "no shared model-specific code"
# directive.
#
# `MelitzCCBundle` is written duck-typed-compatible with every EXISTING Melitz consumer
# function that takes a bundle argument (`direct_gradient.jl`, `sorted_crossing_gradient.jl`,
# `inner_screening.jl`, ...) -- see docs/melitz_production_port_handoff_2026-07-26.md Section
# 4's own "zero function signatures type-annotate `obj`" finding. Consumers that read the
# DENSE `obj.H` directly (not just call the functor) get a new, MORE SPECIFIC method here
# (Julia multiple dispatch on `obj::MelitzCCBundle`) rather than a generic-code change -- the
# original dense method is untouched and remains the path for `PsiObjectiveBundleDelta`/
# `PsiObjectiveBundleImplicit`.

using KNITRO
using LinearAlgebra: BLAS

# ============================================================================
# Melitz-owned single-flight inner-solve guard (ports cc_algo/parallelism_guards.jl's
# guard_enter_inner_solve!/guard_exit_inner_solve! -- same simple invariant, independent
# state, per the user's "Melitz needs its OWN copy of this guard too" directive). Melitz's
# existing coordinate-probe-pool guards (direct_gradient.jl etc.) still call INTO
# cc_algo's own guard_enter_coord_pool!/guard_exit_coord_pool! (pre-existing code, untouched
# by this session) -- those check against cc_algo's OWN INNER_SOLVE_ACTIVE flag, which this
# independent guard does not set. This is a deliberate, narrow, documented scope limitation
# (not a correctness gap in the sequencing this codebase actually exercises: a coordinate
# probe never itself launches an inner KNITRO solve, on either bundle type), not an oversight.
# ============================================================================

const MELITZ_CC_GUARD_ENABLED = Ref(true)
# 2026-07-26 closure session (governing prompt Phase 11): `Threads.Atomic{Bool}`, not a plain
# `Ref(false)` -- audited and found a genuine (if narrow) TOCTOU race in the ORIGINAL
# plain-`Ref` version (mirrored, at the time, from `cc_algo/parallelism_guards.jl`'s own
# identical plain-`Ref` pattern -- that shared file is NOT touched here, per the Ricardian
# ownership boundary; only this Melitz-owned copy is hardened): two threads racing the
# check-then-set on a plain `Ref{Bool}` could BOTH observe `false` before either sets `true`,
# letting a genuine concurrent-KN_solve violation slip past the guard undetected. Under this
# codebase's own standing rule ("never spawn concurrent KN_solve across Threads.@threads",
# confirmed nowhere in this file's actual parallel regions -- the `Threads.@threads`
# coordinate sweeps in `direct_gradient.jl`/`sorted_crossing_gradient.jl`/`moment_operator.jl`
# never call into this guard at all, only ever doing arithmetic on an already-converged dual --
# this guard's exposure is latent, not actively triggered today) this was never observed to
# fire incorrectly, but a defensive guard whose OWN check can race is a real gap worth closing
# now that it's been found, not merely a stylistic preference.
const MELITZ_CC_INNER_SOLVE_ACTIVE = Threads.Atomic{Bool}(false)
const MELITZ_CC_GUARD_VIOLATIONS = Ref(0)

# Melitz-owned KKT diagnostics capture (mirrors cc_algo/inner_loop_functions.jl's
# INNER_LAST_OPT_ERR/INNER_LAST_FEAS_ERR -- independent globals, set by
# melitz_cc_inner_loop_knitro! after every real KNITRO solve, never read from cc_algo's own).
const MELITZ_CC_LAST_OPT_ERR = Ref(NaN)
const MELITZ_CC_LAST_FEAS_ERR = Ref(NaN)

"Call at the very start of every real Melitz-owned inner-KNITRO-solve entry point, before KN_new."
function melitz_cc_guard_enter_inner_solve!()
    MELITZ_CC_GUARD_ENABLED[] || return nothing
    # Atomic compare-and-swap: only ONE caller can ever observe `was_active==false` and
    # thereby "win" the guard, even under genuine concurrent calls -- unlike the prior plain
    # `Ref{Bool}` check-then-set, this cannot let two threads both believe they acquired it.
    was_active = Threads.atomic_cas!(MELITZ_CC_INNER_SOLVE_ACTIVE, false, true)
    if was_active
        MELITZ_CC_GUARD_VIOLATIONS[] += 1
        error("melitz cc_bundle guard: a Melitz-owned inner KNITRO solve was launched while " *
              "another was already active -- concurrent matrix-free inner solves on the same " *
              "bundle are not supported.")
    end
    return nothing
end
"Call in a `finally` clause paired with every melitz_cc_guard_enter_inner_solve! call."
function melitz_cc_guard_exit_inner_solve!()
    MELITZ_CC_GUARD_ENABLED[] || return nothing
    MELITZ_CC_INNER_SOLVE_ACTIVE[] = false
    return nothing
end
"Reset guard state between independent test runs (mirrors cc_algo's guard_reset!/psi_callback_guard_reset!)."
function melitz_cc_guard_reset!()
    MELITZ_CC_INNER_SOLVE_ACTIVE[] = false
    MELITZ_CC_GUARD_VIOLATIONS[] = 0
    return nothing
end

# ============================================================================
# Melitz-owned divergence conjugate -- copied VERBATIM from cc_algo/Psi.jl's Psi!/dPsi!/
# ddPsi! (byte-for-byte identical arithmetic to matrix_free_dual_solve.jl's own `_mf_Psi!`
# family), under permanent, non-underscore-prefixed production names per the 2026-07-26
# handoff's own recommendation ("a production divergence function likely wants a public
# name").
# ============================================================================

function melitz_cc_Psi!(arg1::AbstractVector{Float64}, arg0::AbstractVector{Float64})
    @inbounds for i in 1:length(arg0)
        if arg0[i] <= 1.0
            arg1[i] = exp(arg0[i])
        else
            arg1[i] = arg0[i]^2 + 1.0
            arg1[i] *= 0.5 * exp(1)
        end
    end
    arg1 .-= 1.0
    return arg1
end

function melitz_cc_dPsi!(arg1::AbstractVector{Float64}, arg0::AbstractVector{Float64})
    @inbounds for i in 1:length(arg0)
        arg1[i] = arg0[i] <= 1.0 ? exp(arg0[i]) : exp(1) * arg0[i]
    end
    return arg1
end

function melitz_cc_ddPsi!(arg1::AbstractVector{Float64}, arg0::AbstractVector{Float64})
    @inbounds for i in 1:length(arg0)
        arg1[i] = arg0[i] <= 1.0 ? exp(arg0[i]) : exp(1)
    end
    return arg1
end

# ============================================================================
# MelitzCCBundle: the permanent, Melitz-owned, matrix-free production bundle.
# ============================================================================

"""
    MelitzCCBundle

Permanent Melitz-owned replacement for both `PsiObjectiveBundleDelta` (`mode=:delta`) and
`PsiObjectiveBundleImplicit` (`mode=:implicit`), backed entirely by `MelitzMomentOperator`
(`moment_operator.jl`) -- no `W x (D^2+1)` dense `G`/`H` matrix anywhere in this struct.
Written duck-typed-compatible with every existing Melitz consumer function (see this file's
header): `γ`, `U`, `d`, `outer_constr_index`, `find_smallest`, `lower_limit`, `use_cached_x`,
`x`, `Psi!`, and the `threshold_crossed`/`threshold_crossing_*` instrumentation fields all
match `PsiObjectiveBundleImplicit`'s own field NAMES and semantics exactly, so any consumer
that reads them via plain field access (not `obj.H`) works unchanged.

`op` must be updated at the CURRENT outer point (`melitz_update_operator_at_theta!`) before
the functor/inner-solve driver is called -- mirrors `PsiObjectiveBundleDelta`/`Implicit`'s
own `obj.moments!` being called once per `inner_loop_internal`, before KNITRO starts
iterating, never inside a callback.

`hessian_backend` (`:structured_serial` or `:structured_parallel`, resolved from
`MelitzBackendConfig` at construction time -- never `:auto` on the struct itself) selects
between `melitz_full_weighted_gram!`/`melitz_full_weighted_gram_parallel!` inside the
functor's Hessian branch.
"""
mutable struct MelitzCCBundle
    op::MelitzMomentOperator
    mode::Symbol                 # :delta or :implicit
    γ::Any                       # ctx NamedTuple -- same field name/role as the dense bundles
    U::Matrix{Float64}           # z_draws -- kept for duck-typed consumers reading obj.U
    M::Int
    d::Int
    outer_constr_index::Int
    find_smallest::Bool
    lower_limit::Float64
    policy::MelitzInnerSolvePolicy
    use_cached_x::Bool
    x::Vector{Float64}
    H_save::Float64              # :implicit-mode only: (gamma_prime_j - 1) * (-1)^find_smallest
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    arg2::Vector{Float64}
    Hfull::Matrix{Float64}
    Psi!::Function
    dPsi!::Function
    ddPsi!::Function
    inner_loop_opt::String
    outer_loop_opt::String
    hessian_backend::Symbol
    threshold_crossed::Base.RefValue{Bool}
    threshold_crossing_bound::Base.RefValue{Float64}
    threshold_crossing_x::Base.RefValue{Vector{Float64}}
    threshold_crossing_time_ns::Base.RefValue{UInt64}
    # Fixed (never actually toggled) duck-type-compatibility fields: MelitzCCBundle has NO
    # dense jac_h theta-branch at all (structurally, not merely "currently disabled") -- these
    # two fields exist ONLY so pre-existing consumers/tests that check
    # `obj.needs_outer_moment_jacobian`/`size(obj.jac_h)` (a dense-bundle-only concept) see the
    # truthful answer ("no jac_h tensor exists") rather than a FieldError. Always `false`/
    # `zeros(0,0,0)` -- never set to anything else, since there is no matrix-free equivalent.
    needs_outer_moment_jacobian::Bool
    jac_h::Array{Float64,3}
end

"""
    build_melitz_cc_bundle(op, ctx; mode, U, outer_constr_index, find_smallest=true,
                            policy::MelitzInnerSolvePolicy, inner_loop_opt, outer_loop_opt,
                            hessian_backend=:structured_serial) -> MelitzCCBundle

`op` must already be constructed (`build_melitz_moment_operator`) for the same `(D, W)` as
`U`; its fixed-outer-point fields are left stale until the first
`melitz_update_operator_at_theta!` call (mirrors `MelitzMomentOperator`'s own construction
contract).

**`policy::MelitzInnerSolvePolicy` has NO default -- it is a required keyword argument,
deliberately** (2026-07-28 inner-solver architecture-consolidation session, superseding this
docstring's own prior `lower_limit::Float64` description). `lower_limit` (the evaluation-cap
early-abort threshold: `(Q::MelitzCCBundle)`'s own functor, `if f <= Q.lower_limit; return
-KN_INFINITY` -- telling KNITRO's inner solve to stop immediately once a point is certifiably
bad) is now DERIVED from `policy` (`melitz_policy_lower_limit(policy)`, `inner_solve_policy.jl`)
rather than accepted as an independent `Float64` -- the bundle's own `policy` field
(`melitz_apply_policy_to_knitro!`, `melitz_cc_inner_loop_knitro!` below) is also the SAME
object `melitz_classified_inner_solve`'s successor, `_melitz_classified_inner_solve!`
(`inner_screening.jl`), reads its cap from via the wrapping `MelitzInnerSession` -- there is
no second, independently-suppliable cap value anywhere in this bundle's lifecycle.

This codebase's own recorded history (see this repo's `CLAUDE.md` and the memory this exact
class of mistake generated) shows the evaluation cap has been SILENTLY left inactive,
repeatedly, across multiple sessions, because a previous `lower_limit::Float64` REQUIRED
keyword (itself a fix for an even earlier silent default) was still just a bare number that a
caller three levels up could compute incorrectly or omit propagating (confirmed live as the
root cause of the `1.510118e14` `FiniteSolved`-above-cap anomaly,
`docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md`: a wrapper
translated a missing config into `-KNITRO.KN_INFINITY` one layer up from this constructor).
Requiring a `policy::MelitzInnerSolvePolicy` object here instead of a raw `Float64` closes
that: `CappedEvaluation`/`FullValueEvaluation` (`inner_solve_policy.jl`) are the only two ways
to construct one, both validating at construction, and the object itself (not a derived
number) is what every downstream consumer (this constructor, the KNITRO-instance policy
application below, the classifier's cap-derivation) reads from -- there is no intermediate
step where a caller computes or forgets to pass the RIGHT number. Callers that deliberately
want no cap must pass `policy=FullValueEvaluation()` EXPLICITLY -- a conscious, visible,
NAMED choice at the call site, not a silent inherited default or a bare `-Inf`.
"""
function build_melitz_cc_bundle(op::MelitzMomentOperator, ctx;
                                 mode::Symbol, U::AbstractMatrix,
                                 outer_constr_index::Int=op.layout.num_moments + 1,
                                 find_smallest::Bool=true,
                                 policy::MelitzInnerSolvePolicy,
                                 inner_loop_opt::AbstractString,
                                 outer_loop_opt::AbstractString,
                                 hessian_backend::Symbol=:structured_serial)
    mode in (:delta, :implicit) || throw(ArgumentError("build_melitz_cc_bundle: mode must be :delta or :implicit, got $mode"))
    hessian_backend in (:structured_serial, :structured_parallel) || throw(ArgumentError(
        "build_melitz_cc_bundle: hessian_backend must be :structured_serial or :structured_parallel, got $hessian_backend"))
    W = op.W
    d = op.layout.num_moments
    lower_limit = melitz_policy_lower_limit(policy)
    return MelitzCCBundle(op, mode, ctx, Matrix{Float64}(U), W, d, outer_constr_index,
        find_smallest, lower_limit, policy, false, fill(NaN, outer_constr_index), 0.0,
        zeros(W), zeros(W), zeros(W), zeros(d + 1, d + 1),
        melitz_cc_Psi!, melitz_cc_dPsi!, melitz_cc_ddPsi!,
        String(inner_loop_opt), String(outer_loop_opt), hessian_backend,
        Ref(false), Ref(NaN), Ref(Float64[]), Ref(UInt64(0)),
        false, zeros(Float64, 0, 0, 0))
end

@inline function _melitz_op_gbase(op::MelitzMomentOperator, o::Int, d::Int, w::Int)
    active = op.bin[w, o] >= op.rank[d, o]
    return active ? op.coef[o, d] * op.sorted_ctx.z_power_original[w, o] - op.lambda[o, d] : -op.lambda[o, d]
end

# ============================================================================
# Functor: matches PsiObjectiveBundleImplicit/Delta's own calling convention
# `(x, g=Float64[]; h=Float64[])` for the fixed-theta inner CC dual solve. The `(x, g, θ;
# jac=...)` theta-gradient branch is DELIBERATELY NOT IMPLEMENTED here -- confirmed
# unreachable for both bundle types on the production-fast path: the default
# `outer_gradient_backend` (Phase 3) is always one of the "direct"/"sorted" families, which
# bypass this functor's theta-branch entirely (finite_delta_outer.jl's `cb_G!`, `is_direct`
# branch); `needs_outer_moment_jacobian`'s dense-bundle analogue confirms this branch is
# vacuous for Melitz's actual outer_constr_index==d+1 convention (2026-07-26 handoff Section
# 5). Throws a clear error if a caller ever passes a nonempty `θ`, rather than silently
# misbehaving.
# ============================================================================

function (Q::MelitzCCBundle)(x::AbstractVector{Float64}, g::AbstractVector{Float64}=Float64[],
                              θ::AbstractVector{Float64}=Float64[]; h::AbstractVector{Float64}=Float64[],
                              constr::AbstractVector{Float64}=Float64[],
                              jac::AbstractMatrix{Float64}=Array{Float64}(undef, 0, 0))
    length(θ) == 0 || error("MelitzCCBundle functor: the outer theta-gradient branch (jac_h) " *
        "is not implemented -- confirmed unreachable on the production-fast path (see this " *
        "file's header). Use gradient_backend in the :B_direct_argument_* family.")
    zeta = x[1]
    mu = @view x[2:end]

    # Governing prompt Phase 2: this functor is KNITRO's own registered objective/gradient/
    # Hessian callback for the inner dual solve -- called many times per single FC's inner
    # KNITRO solve (every Newton iteration), so these are the direct, per-callback-body
    # timers Phase 2 asks for ("inner objective/gradient/Hessian callback wall"), as opposed
    # to the coarser whole-inner-solve wrapper timing that already existed
    # (`:inner_solve_warm_success` etc., finite_delta_outer.jl). `@melitz_profile`'s own
    # `time_ns()` overhead (~tens of ns) is negligible next to the real compute cost measured
    # here even at this call frequency, and is exactly zero when MELITZ_PROFILE[] is off.
    @melitz_profile :fc_inner_obj_eval begin
        mul_G!(Q.arg0, Q.op, zeta, mu)
        melitz_cc_Psi!(Q.arg1, Q.arg0)
    end
    f = sum(Q.arg1) / Q.M + zeta
    MELITZ_MATRIX_FREE_OBJECTIVE_CALLS[] += 1

    if length(g) > 0 || length(h) > 0 || length(constr) > 0
        @melitz_profile :fc_inner_dpsi_eval melitz_cc_dPsi!(Q.arg1, Q.arg0)
    end

    if length(constr) > 0
        # Matches PsiObjectiveBundleImplicit's own functor EXACTLY: `constr[1] = -f*1e10`
        # unconditionally, then an "extra outer-constraint moments" BLAS.gemv! ONLY IF
        # `outer_constr_index <= d` -- structurally FALSE for Melitz (outer_constr_index ==
        # d+1 always, 2026-07-26 handoff Section 5's confirmed finding), so that second part
        # is genuinely vacuous and correctly omitted here (not the whole `constr` branch,
        # which IS live: finite_delta_outer.jl's cb_F! calls `obj(x, constr=local_c)` to
        # recover `Delta_theta = local_c[1]/1e10` from the just-converged dual).
        constr[1] = -f * 1e10
    end

    if length(g) > 0
        @melitz_profile :fc_inner_grad_eval begin
            g[1] = 1.0 - sum(Q.arg1) / Q.M
            gmu = @view g[2:end]
            mul_Gt!(gmu, Q.op, Q.arg1)
            gmu .*= -1.0 / Q.M
        end
        MELITZ_MATRIX_FREE_GRADIENT_CALLS[] += 1
    end

    if length(h) > 0
        @melitz_profile :fc_inner_hess_eval begin
            melitz_cc_ddPsi!(Q.arg2, Q.arg0)
            if Q.hessian_backend == :structured_parallel
                melitz_full_weighted_gram_parallel!(Q.Hfull, Q.op, Q.arg2)
            else
                melitz_full_weighted_gram!(Q.Hfull, Q.op, Q.arg2)
            end
            n = Q.d + 1
            invM = 1.0 / Q.M
            k = 1
            @inbounds for i in 1:n
                for j in i:n
                    h[k] = Q.Hfull[i, j] * invM
                    k += 1
                end
            end
        end
        MELITZ_MATRIX_FREE_HESSIAN_CALLS[] += 1
    end

    if f <= Q.lower_limit
        if Q.mode == :implicit
            Q.threshold_crossed[] = true
            Q.threshold_crossing_bound[] = -f
            Q.threshold_crossing_x[] = copy(x)
            Q.threshold_crossing_time_ns[] = time_ns()
        end
        MELITZ_EVALUATION_CAP_EXITS[] += 1
        return -KNITRO.KN_INFINITY
    else
        return f
    end
end

# ============================================================================
# theta -> operator update (ports melitz_moments_adapter!'s expand-theta/re-equilibration
# logic -- SAME algebra, reused verbatim, writing into a MelitzMomentOperator instead of a
# dense (K, G) pair). Returns gamma_prime_j (the scalar `:implicit`-mode K value needs).
# ============================================================================

"""
    melitz_update_operator_at_theta!(op, theta, ctx) -> gamma_prime_j::Float64

Matrix-free analogue of `melitz_moments_adapter!` (delta_star.jl): re-equilibrates
`(A, f, gamma_prime_target)` and the autarky counterfactual from the FREE outer vector
`theta` (exactly the same `melitz_expand_theta`/`melitz_baseline_cutoff` calls, same
`MelitzPrimitives`/`MelitzEquilibrium`/`MelitzCounterfactual` construction), then updates
`op` in place via `melitz_update_moment_operator!` instead of building a dense `(K, G)` pair.
Returns `gamma_prime_j` so the caller can set `H_save = (gamma_prime_j - 1) * (-1)^find_smallest`
for `:implicit`-mode bundles (mirrors `melitz_moments_adapter!`'s own `K .=
p.gamma_prime_target - 1`).
"""
function melitz_update_operator_at_theta!(op::MelitzMomentOperator, theta::AbstractVector, ctx)
    # Governing prompt Phase 2 (2026-07-XX outer-search session): split this function's own
    # two economically distinct steps -- theta expansion (gravity-pivot reconstruction, O(D^2))
    # and the moment-operator merge sweep (O(W*D)) -- into separate `@melitz_profile`
    # categories, since a prior closure session's own audit (docs/melitz_final_allocation_and_
    # gradient_closure_2026-07-27.md Phase 1.4) measured this whole function at ~2KB/32KB
    # (D=4/real D=20) but never isolated which of the two sub-steps a real FC's wall-time
    # actually goes to. Zero-cost when MELITZ_PROFILE[] is off (the macro's own guarantee).
    A, f, gamma_prime_j, f_jj = @melitz_profile :fc_theta_expand melitz_expand_theta(theta, ctx)
    D = ctx.D
    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country,
                                   ctx.tau, ctx.w, A, f, gamma_prime_j)
    cutoff = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    eq = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), cutoff, ctx.X_data)
    expenditure_prime = ctx.w_prime * ctx.L[ctx.target_country]
    cf = MelitzCounterfactual(ctx.target_country, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)
    @melitz_profile :fc_operator_merge melitz_update_moment_operator!(op, primitives, eq, cf; X_data=ctx.X_data)
    MELITZ_OPERATOR_REBUILDS[] += 1
    return gamma_prime_j
end

# ============================================================================
# Melitz-owned inner KNITRO driver -- ports melitz_matrix_free_inner_solve
# (matrix_free_dual_solve.jl) into the permanent bundle, folding in the find_smallest
# sign-wrapper convention and the per-mode inner_loop_internal return-value convention
# (mirrors cc_algo/inner_loop_functions.jl's inner_loop/inner_loop_internal EXACTLY --
# ported, not called).
# ============================================================================

"""
    melitz_cc_inner_loop_knitro!(bundle::MelitzCCBundle) -> (nStatus, objSol, x)

Drives one real KNITRO solve of the fixed-theta inner CC dual program against `bundle`'s
CURRENT operator state (`op` must already reflect the desired theta --
`melitz_update_operator_at_theta!`, called by `melitz_cc_inner_loop_internal!` below, never
inside this function). Mirrors `inner_loop_KNITRO`'s exact API call sequence (same KNITRO.jl
calls, same options-file loading, same Hessian-callback gating on `hessopt`).
"""
function melitz_cc_inner_loop_knitro!(bundle::MelitzCCBundle)
    melitz_cc_guard_enter_inner_solve!()
    try
        n = bundle.outer_constr_index
        kc = KNITRO.KN_new()

        melitz_kn_add_vars!(kc, n)
        KNITRO.KN_set_var_lobnds_all(kc, fill(-KNITRO.KN_INFINITY, n))
        x0 = (bundle.use_cached_x && all(isfinite, bundle.x)) ? bundle.x : zeros(n)
        KNITRO.KN_set_var_primal_init_values_all(kc, x0)

        cbEvalFG! = (kc2, cb2, evalRequest, evalResult, userParams) -> begin
            evalResult.obj[1] = bundle(evalRequest.x, evalResult.objGrad)
            return 0
        end
        cbEvalH! = (kc2, cb2, evalRequest, evalResult, userParams) -> begin
            bundle(evalRequest.x, Float64[]; h=evalResult.hess)
            return 0
        end

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cbEvalFG!)
        KNITRO.KN_load_param_file(kc, bundle.inner_loop_opt)
        # 2026-07-28 inner-solver architecture-consolidation session (governing prompt
        # Section 3): apply bundle.policy's own max_iterations/max_seconds directly to this
        # KNITRO instance, AFTER the static .opt file load, so the policy's values win. Only
        # done here, on the Melitz-owned matrix-free driver -- the legacy dense bundles'
        # KNITRO instance is constructed inside cc_algo/inner_loop_functions.jl (Ricardian,
        # never touched by this session); see CappedEvaluation's own docstring
        # (inner_solve_policy.jl) for that disclosed limitation.
        melitz_apply_policy_to_knitro!(kc, bundle.policy)
        if melitz_kn_get_int_param(kc, "hessopt") == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, cbEvalH!)
        end

        KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        opt_err_ref = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc, opt_err_ref)
        feas_err_ref = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_feas_error(kc, feas_err_ref)
        MELITZ_CC_LAST_OPT_ERR[] = Float64(opt_err_ref[])
        MELITZ_CC_LAST_FEAS_ERR[] = Float64(feas_err_ref[])
        KNITRO.KN_free(kc)

        if nStatus in (0, -100, -101, -103)
            bundle.x .= x
        end
        return nStatus, objSol, x
    finally
        melitz_cc_guard_exit_inner_solve!()
    end
end

"""
    melitz_cc_inner_loop_internal!(obj::MelitzCCBundle, theta) -> (val, x, nStatus)

Melitz-owned port of `cc_algo/inner_loop_functions.jl`'s `inner_loop_internal`, dispatched
per `obj.mode` exactly as the generic dense function dispatches per bundle TYPE
(`PsiObjectiveBundleDelta` vs `PsiObjectiveBundleImplicit`): updates the operator at `theta`
(`melitz_update_operator_at_theta!`), runs the real KNITRO solve, and returns the raw
objective (`mode=:delta`) or `H_save` (`mode=:implicit`, `(gamma_prime_j-1)*(-1)^find_smallest`
-- exactly `inner_loop_internal(obj::PsiObjectiveBundleImplicit,...)`'s own convention) on
acceptance, or `-1e10`/`NaN` dual on rejection (matching the dense reference's own
`obj.x .= NaN` / `-1e10` sentinel).
"""
function melitz_cc_inner_loop_internal!(obj::MelitzCCBundle, theta::AbstractVector)
    gamma_prime_j = melitz_update_operator_at_theta!(obj.op, theta, obj.γ)
    if obj.mode == :implicit
        obj.H_save = (gamma_prime_j - 1.0) * (-1.0)^obj.find_smallest
    end

    nStatus, objSol, x = melitz_cc_inner_loop_knitro!(obj)

    if obj.mode == :implicit
        if nStatus in (0, -100, -101, -103)
            obj.x .= x
            return obj.H_save, x, nStatus
        else
            obj.x .= NaN
            MELITZ_NUMERICAL_FAILURES[] += 1
            return -1e10, x, nStatus
        end
    else   # :delta
        if nStatus in (0, -100, -101, -103)
            obj.x .= x
            return objSol, x, nStatus
        else
            obj.x .= NaN
            MELITZ_NUMERICAL_FAILURES[] += 1
            return -1e10, x, nStatus
        end
    end
end

"""
    melitz_cc_inner_loop(obj::MelitzCCBundle, theta) -> (val, x, nStatus)

Melitz-owned port of the generic `inner_loop` WRAPPER (find_smallest sign flip on top of
`inner_loop_internal`) -- used ONLY for `mode=:delta` bundles (mirrors the dense reference's
own usage: `melitz_recover_lfd`/`run_melitz_inner_delta` call generic `inner_loop`, never
`inner_loop_internal` directly; `mode=:implicit` production consumers
(`melitz_classified_inner_solve`) call `melitz_cc_inner_loop_internal!` directly instead,
exactly as they call `CS.inner_loop_internal` directly for the dense Implicit bundle).
"""
function melitz_cc_inner_loop(obj::MelitzCCBundle, theta::AbstractVector)
    val, x, nStatus = melitz_cc_inner_loop_internal!(obj, theta)
    obj.find_smallest && (val *= -1.0)
    return val, x, nStatus
end

# ============================================================================
# Dispatch-based "duck typing extension" -- new, MORE SPECIFIC methods for functions defined
# elsewhere (direct_gradient.jl, sorted_crossing_gradient.jl, delta_star.jl) that read the
# dense `obj.H` directly, keyed on `obj::MelitzCCBundle`. The original, untouched generic
# methods remain the path for `PsiObjectiveBundleDelta`/`PsiObjectiveBundleImplicit`.
# ============================================================================

"_base_arg0! (direct_gradient.jl) for the matrix-free bundle: the SAME `u = -zeta - G*mu` quantity, via mul_G! instead of BLAS.gemv! on obj.H."
function _base_arg0!(arg0_base::AbstractVector{Float64}, obj::MelitzCCBundle, x::AbstractVector{Float64})
    mul_G!(arg0_base, obj.op, x[1], @view(x[2:end]))
    return arg0_base
end

"""
_direct_coordinate_grad (direct_gradient.jl) for the matrix-free bundle: identical formula,
`gbase` reconstructed from operator fields (`_melitz_op_gbase`/`op.ell`) instead of read from
`obj.H`.
"""
function _direct_coordinate_grad(cc::MelitzCompactColumns, theta_p::AbstractVector{Float64},
                                  theta_m::AbstractVector{Float64}, ctx, obj::MelitzCCBundle,
                                  lambda::AbstractVector{Float64}, arg0_base::AbstractVector{Float64},
                                  h::Real, Gp::AbstractMatrix{Float64}, Gm::AbstractMatrix{Float64},
                                  linkp::AbstractVector{Float64}, linkm::AbstractVector{Float64},
                                  profit::AbstractVector{Float64}, u_plus::AbstractVector{Float64},
                                  u_minus::AbstractVector{Float64}, psi_buf::AbstractVector{Float64},
                                  state_p::MelitzExpandedState, state_m::MelitzExpandedState,
                                  ws::MelitzThetaExpansionWorkspace)
    W = length(arg0_base)
    layout = ctx.moment_layout
    ncols = length(cc.direct_cols)
    op = obj.op

    # 2026-07-27 continuation (governing prompt Phase 1.3, this session): see
    # direct_gradient.jl's identical fix -- state_p/state_m expanded once per coordinate probe,
    # shared by the direct-column fill and the focal-link fill, for the MelitzCCBundle-specific
    # method of this (non-sorted) direct backend.
    melitz_expand_theta!(state_p, theta_p, ctx, ws)
    melitz_expand_theta!(state_m, theta_m, ctx, ws)

    ncols > 0 && _fill_compact_direct_columns_from_state!(Gp, ctx, obj, state_p, cc.direct_cells, ncols)
    ncols > 0 && _fill_compact_direct_columns_from_state!(Gm, ctx, obj, state_m, cc.direct_cells, ncols)
    if cc.touches_link
        _fill_compact_link_from_state!(linkp, profit, ctx, obj, state_p)
        _fill_compact_link_from_state!(linkm, profit, ctx, obj, state_m)
    end

    copyto!(u_plus, arg0_base)
    copyto!(u_minus, arg0_base)
    @inbounds for idx in 1:ncols
        gcol = cc.direct_cols[idx]
        lam_k = lambda[gcol]
        (o, d) = cc.direct_cells[idx]
        for w in 1:W
            gbase = _melitz_op_gbase(op, o, d, w)
            u_plus[w]  -= lam_k * (Gp[w, idx] - gbase)
            u_minus[w] -= lam_k * (Gm[w, idx] - gbase)
        end
    end
    if cc.touches_link
        lam_link = lambda[layout.focal_link_index]
        ell = op.ell
        @inbounds for w in 1:W
            gbase = ell[w]
            u_plus[w]  -= lam_link * (linkp[w] - gbase)
            u_minus[w] -= lam_link * (linkm[w] - gbase)
        end
    end

    melitz_cc_Psi!(psi_buf, u_plus)
    L_plus = sum(psi_buf) / W
    melitz_cc_Psi!(psi_buf, u_minus)
    L_minus = sum(psi_buf) / W

    return -1e10 * (L_plus - L_minus) / (2h)
end

"""
_direct_coordinate_grad_sorted (sorted_crossing_gradient.jl) for the matrix-free bundle:
same crossing-slice restriction (only `union_start[idx]:W` per touched column), `gbase`
reconstructed from operator fields instead of `obj.H`.
"""
function _direct_coordinate_grad_sorted(cc::MelitzCompactColumns, theta_p::AbstractVector{Float64},
                                         theta_m::AbstractVector{Float64}, ctx, obj::MelitzCCBundle,
                                         sorted_ctx::MelitzSortedTailContext, lambda::AbstractVector{Float64},
                                         arg0_base::AbstractVector{Float64}, h::Real,
                                         Gp::AbstractMatrix{Float64}, Gm::AbstractMatrix{Float64},
                                         union_start::AbstractVector{Int}, linkp::AbstractVector{Float64},
                                         linkm::AbstractVector{Float64}, profit::AbstractVector{Float64},
                                         u_plus::AbstractVector{Float64}, u_minus::AbstractVector{Float64},
                                         psi_buf::AbstractVector{Float64}, state_p::MelitzExpandedState,
                                         state_m::MelitzExpandedState, ws::MelitzThetaExpansionWorkspace)
    W = length(arg0_base)
    layout = ctx.moment_layout
    ncols = length(cc.direct_cols)
    op = obj.op

    copyto!(u_plus, arg0_base)
    copyto!(u_minus, arg0_base)

    # 2026-07-27 continuation (governing prompt Phase 1.1, this session): see
    # sorted_crossing_gradient.jl's identical fix/comment -- state_p/state_m are expanded
    # ONCE per coordinate probe here (MelitzCCBundle's own copy of this function), shared by
    # the direct-column fill and the focal-link fill, eliminating the prior session's
    # remaining allocating melitz_expand_theta call inside _fill_compact_link!.
    melitz_expand_theta!(state_p, theta_p, ctx, ws)
    melitz_expand_theta!(state_m, theta_m, ctx, ws)

    if ncols > 0
        _fill_compact_direct_columns_crossing_sorted!(Gp, Gm, union_start, ctx, sorted_ctx,
            cc.direct_cells, ncols, state_p, state_m)
        @inbounds for idx in 1:ncols
            gcol = cc.direct_cols[idx]
            lam_k = lambda[gcol]
            (o, d) = cc.direct_cells[idx]
            perm_o = @view sorted_ctx.permutation[:, o]
            kunion = union_start[idx]
            for pos in kunion:W
                w = perm_o[pos]
                gbase = _melitz_op_gbase(op, o, d, w)
                u_plus[w] -= lam_k * (Gp[w, idx] - gbase)
                u_minus[w] -= lam_k * (Gm[w, idx] - gbase)
            end
        end
    end

    if cc.touches_link
        _fill_compact_link_from_state!(linkp, profit, ctx, obj, state_p)
        _fill_compact_link_from_state!(linkm, profit, ctx, obj, state_m)
        lam_link = lambda[layout.focal_link_index]
        ell = op.ell
        @inbounds for w in 1:W
            gbase = ell[w]
            u_plus[w] -= lam_link * (linkp[w] - gbase)
            u_minus[w] -= lam_link * (linkm[w] - gbase)
        end
    end

    melitz_cc_Psi!(psi_buf, u_plus)
    L_plus = sum(psi_buf) / W
    melitz_cc_Psi!(psi_buf, u_minus)
    L_minus = sum(psi_buf) / W

    return -1e10 * (L_plus - L_minus) / (2h)
end

"""
_direct_coordinate_grad_touched_row (touched_row_gradient.jl) for the matrix-free bundle:
same touched-row-only accumulate/evaluate as the generic method, `gbase` reconstructed from
operator fields (`_melitz_op_gbase`/`op.ell`) instead of `obj.H`, mirroring
`_direct_coordinate_grad_sorted`'s own MelitzCCBundle-specific method immediately above.
"""
function _direct_coordinate_grad_touched_row(cc::MelitzCompactColumns, theta_p::AbstractVector{Float64},
                                              theta_m::AbstractVector{Float64}, ctx, obj::MelitzCCBundle,
                                              sorted_ctx::MelitzSortedTailContext, lambda::AbstractVector{Float64},
                                              arg0_base::AbstractVector{Float64}, psi_base::AbstractVector{Float64},
                                              base_scalar_sum::Float64, h::Real,
                                              Gp::AbstractMatrix{Float64}, Gm::AbstractMatrix{Float64},
                                              union_start::AbstractVector{Int}, linkp::AbstractVector{Float64},
                                              linkm::AbstractVector{Float64}, profit::AbstractVector{Float64},
                                              delta_plus::Vector{Float64}, delta_minus::Vector{Float64},
                                              touched_gen::Vector{Int}, gen::Int, touched_list::Vector{Int},
                                              state_p::MelitzExpandedState, state_m::MelitzExpandedState,
                                              ws::MelitzThetaExpansionWorkspace)
    W = length(arg0_base)
    layout = ctx.moment_layout
    ncols = length(cc.direct_cols)
    op = obj.op
    empty!(touched_list)

    melitz_expand_theta!(state_p, theta_p, ctx, ws)
    melitz_expand_theta!(state_m, theta_m, ctx, ws)

    if ncols > 0
        _fill_compact_direct_columns_crossing_sorted!(Gp, Gm, union_start, ctx, sorted_ctx,
            cc.direct_cells, ncols, state_p, state_m)
        @inbounds for idx in 1:ncols
            gcol = cc.direct_cols[idx]
            lam_k = lambda[gcol]
            (o, d) = cc.direct_cells[idx]
            perm_o = @view sorted_ctx.permutation[:, o]
            kunion = union_start[idx]
            for pos in kunion:W
                w = perm_o[pos]
                gbase = _melitz_op_gbase(op, o, d, w)
                cp = -lam_k * (Gp[w, idx] - gbase)
                cm = -lam_k * (Gm[w, idx] - gbase)
                _touch_row!(w, cp, cm, delta_plus, delta_minus, touched_gen, gen, touched_list)
            end
        end
    end

    if cc.touches_link
        _fill_compact_link_from_state!(linkp, profit, ctx, obj, state_p)
        _fill_compact_link_from_state!(linkm, profit, ctx, obj, state_m)
        lam_link = lambda[layout.focal_link_index]
        ell = op.ell
        @inbounds for w in 1:W
            gbase = ell[w]
            cp = -lam_link * (linkp[w] - gbase)
            cm = -lam_link * (linkm[w] - gbase)
            _touch_row!(w, cp, cm, delta_plus, delta_minus, touched_gen, gen, touched_list)
        end
    end

    scalar_plus = 0.0
    scalar_minus = 0.0
    @inbounds for w in touched_list
        up_val = arg0_base[w] + delta_plus[w]
        um_val = arg0_base[w] + delta_minus[w]
        scalar_plus += _touched_row_psi_scalar(up_val) - psi_base[w]
        scalar_minus += _touched_row_psi_scalar(um_val) - psi_base[w]
    end
    L_plus = (base_scalar_sum + scalar_plus) / W
    L_minus = (base_scalar_sum + scalar_minus) / W

    return -1e10 * (L_plus - L_minus) / (2h)
end

"""
melitz_recover_lfd_from_solution (delta_star.jl) for the matrix-free bundle: SAME LFD
recovery/verification formulas, via `mul_G!`/`mul_Gt!` instead of a dense `G` (Phase 6.4 --
"never index a dense obj.H" / "never materialize G merely to verify a production solve").
`G_precomputed` is accepted for signature compatibility but ignored (a caller passing it for
a MelitzCCBundle is passing dense state that does not apply here); a caller wanting a
verification-only dense cross-check should use the dense-reference bundle instead, or
`melitz_dense_G_from_operator` (diagnostic-only) explicitly.
"""
function melitz_recover_lfd_from_solution(val::Real, x::AbstractVector, nStatus::Integer,
                                           theta::AbstractVector, obj::MelitzCCBundle;
                                           kkt_opt_error::Real=NaN, kkt_feas_error::Real=NaN,
                                           moment_tol::Real=1e-6, normalization_tol::Real=1e-6,
                                           gap_atol::Real=1e-10, gap_rtol::Real=1e-6,
                                           G_precomputed::Union{Nothing,AbstractMatrix}=nothing)
    d = obj.d
    W = obj.op.W

    if nStatus != 0 || !all(isfinite, x)
        return MelitzLFDResult(val, x, nStatus, fill(1.0 / W, W), false,
                                fill(NaN, d), NaN, NaN, NaN, NaN,
                                NaN, val, NaN, NaN, NaN, kkt_opt_error, kkt_feas_error)
    end

    zeta = x[1]
    mu = @view x[2:end]
    # 2026-07-27 addendum (governing prompt Phase 4): `arg0`/`LFD` used to be fresh W-length
    # `zeros(W)` allocations on EVERY call -- the exact class of anti-pattern the addendum
    # flagged (a hot per-callback function allocating several W-length arrays every call),
    # confirmed live via `@allocated`: 640,672 bytes at D=4/W=20,000, scaling to 2,560,672
    # bytes at W=80,000 (clean 4x, i.e. genuinely O(W)), from THIS function alone, called
    # every `cb_F!` via `register_live_candidate!`. Reuses `obj.arg0`/`obj.arg1` instead --
    # SAFE because both are pure functor-internal scratch (grep-confirmed: read/written only
    # inside `(Q::MelitzCCBundle)`'s own callback body, `cc_bundle.jl` above; nothing reads
    # them AFTER a solve completes) and `melitz_recover_lfd_from_solution` always runs strictly
    # AFTER `melitz_cc_inner_loop`'s `KN_solve` has fully finished -- never concurrently with
    # the functor's own use of these same buffers. `weights`/`moment_residuals` themselves
    # remain FRESH allocations (unavoidable: both are stored directly on the returned, often
    # long-lived-cached `MelitzLFDResult`/`MelitzDeltaEvalResult`, so they must be
    # independently owned, not aliased to `obj`'s own mutable scratch).
    arg0 = obj.arg0
    mul_G!(arg0, obj.op, zeta, mu)
    LFD = obj.arg1
    melitz_cc_dPsi!(LFD, arg0)
    s = sum(LFD)

    if !(isfinite(s) && s > 0) || !all(isfinite, LFD) || !all(>=(0), LFD)
        return MelitzLFDResult(val, x, nStatus, fill(1.0 / W, W), false,
                                fill(NaN, d), NaN, NaN, NaN, NaN,
                                NaN, val, NaN, NaN, NaN, kkt_opt_error, kkt_feas_error)
    end

    normalization_residual = s / W - 1
    weights = LFD ./ s
    moment_residuals = zeros(d)
    mul_Gt!(moment_residuals, obj.op, weights)
    max_moment_residual = maximum(abs, moment_residuals)   # fused, no d-length temporary

    primal_divergence = melitz_primal_divergence(weights, W)
    dual_divergence = val
    primal_dual_gap = abs(primal_divergence - dual_divergence)
    gap_tol = max(gap_atol, gap_rtol * max(1.0, abs(primal_divergence), abs(dual_divergence)))

    lfd_ok = abs(normalization_residual) < normalization_tol &&
             max_moment_residual < moment_tol &&
             primal_dual_gap <= gap_tol

    # fused (no `W .* weights .- 1` / `abs.(...)` W-length temporaries)
    max_abs_W_weights_minus_1 = maximum(w -> abs(W * w - 1), weights)

    return MelitzLFDResult(val, x, nStatus, weights, lfd_ok, moment_residuals, normalization_residual,
                            minimum(weights), maximum(weights), max_abs_W_weights_minus_1,
                            primal_divergence, dual_divergence, primal_dual_gap,
                            max_moment_residual, normalization_residual,
                            kkt_opt_error, kkt_feas_error)
end

"""
melitz_recover_lfd (delta_star.jl) for the matrix-free bundle: runs `melitz_cc_inner_loop`
(Melitz-owned, real KNITRO) then delegates to the matrix-free `melitz_recover_lfd_from_solution`
method above.
"""
function melitz_recover_lfd(obj::MelitzCCBundle, theta::AbstractVector; moment_tol::Real=1e-6,
                             normalization_tol::Real=1e-6, gap_atol::Real=1e-10, gap_rtol::Real=1e-6)
    val, x, nStatus = melitz_cc_inner_loop(obj, theta)
    kkt_opt_error = MELITZ_CC_LAST_OPT_ERR[]
    kkt_feas_error = MELITZ_CC_LAST_FEAS_ERR[]
    return melitz_recover_lfd_from_solution(val, x, nStatus, theta, obj;
        kkt_opt_error=kkt_opt_error, kkt_feas_error=kkt_feas_error,
        moment_tol=moment_tol, normalization_tol=normalization_tol, gap_atol=gap_atol, gap_rtol=gap_rtol)
end

# ============================================================================
# Generic Melitz-owned dispatch wrappers used by the production screening/outer-callback
# path (inner_screening.jl, finite_delta_outer.jl) -- ONE new Melitz-owned generic function
# per site that used to hardcode `CS.select_G_from_H`/`obj.moments!`/`CS.inner_loop_internal`,
# with a method for each bundle family. This is the mechanism that lets
# `melitz_classified_inner_solve`/`inner_solve_verified_or_fail`/the exact-point cache work
# unchanged against BOTH bundle types (see Phase 8/9 wiring below).
# ============================================================================

"""
    melitz_bundle_prepare_at_theta!(obj, theta) -> Union{Nothing,AbstractMatrix}

Ensures `obj` reflects `theta` (fills the dense moment matrix for the legacy bundles; updates
the matrix-free operator in place for `MelitzCCBundle`). Returns the dense `G` view for the
legacy bundles (so an existing screen/consumer can keep reading it), or `nothing` for
`MelitzCCBundle` (signals "no dense G exists -- skip any screen that needs one").
"""
function melitz_bundle_prepare_at_theta!(obj, theta::AbstractVector)
    # Generic (untyped) fallback method -- matches this codebase's own established
    # duck-typing convention (2026-07-26 handoff Section 4: "zero function signatures
    # type-annotate obj" across src/melitz/*.jl). Type-annotating this method with
    # `PsiObjectiveBundleDelta`/`PsiObjectiveBundleImplicit` directly would force
    # `cc_algo`/`CounterfactualSensitivity` to be loaded before this FILE could even be
    # included (a type used in a method SIGNATURE must exist at method-definition time,
    # unlike a type used only in a constructor CALL inside a function body) -- breaking this
    # codebase's deliberate cc_algo-independent load order. The specific
    # `obj::MelitzCCBundle` method below is dispatched to first for that type; every other
    # `obj` (in practice, always `PsiObjectiveBundleDelta`/`PsiObjectiveBundleImplicit`)
    # falls through to this one.
    CS = CounterfactualSensitivity
    G_now = CS.select_G_from_H(obj, obj.H)
    obj.moments!(@view(obj.H[:, 1]), G_now, theta, obj.U, obj)
    obj.H[:, 2] .= 1.0
    MELITZ_DENSE_MOMENT_CALLS[] += 1
    MELITZ_DENSE_G_MATERIALIZATIONS[] += 1
    return G_now
end
function melitz_bundle_prepare_at_theta!(obj::MelitzCCBundle, theta::AbstractVector)
    gamma_prime_j = melitz_update_operator_at_theta!(obj.op, theta, obj.γ)
    if obj.mode == :implicit
        obj.H_save = (gamma_prime_j - 1.0) * (-1.0)^obj.find_smallest
    end
    MELITZ_SORTED_MOMENT_CALLS[] += 1
    return nothing
end

"""
    melitz_bundle_inner_solve!(obj, theta) -> (val, x, nStatus)

Runs the fixed-theta inner CC dual solve, dispatched per bundle family -- the legacy bundles
go through `CS.inner_loop_internal` (unmodified cc_algo call, exactly as before this
session); `MelitzCCBundle` goes through the Melitz-owned `melitz_cc_inner_loop_internal!`.
Does NOT re-run `melitz_bundle_prepare_at_theta!` -- the caller is expected to have already
called it at the same `theta` (mirrors the dense reference's own existing, slightly
redundant `CS.inner_loop_internal` re-fill for the legacy bundles, kept unchanged there; the
matrix-free path deliberately avoids the redundant second operator rebuild here since it
would otherwise show up as a spurious `MELITZ_OPERATOR_REBUILDS` count within a single FC
evaluation).
"""
function melitz_bundle_inner_solve!(obj, theta::AbstractVector)
    # Generic (untyped) fallback method -- see melitz_bundle_prepare_at_theta!'s docstring
    # for why this is deliberately untyped rather than Union{PsiObjectiveBundleDelta,...}.
    CS = CounterfactualSensitivity
    return CS.inner_loop_internal(obj, theta)
end
function melitz_bundle_inner_solve!(obj::MelitzCCBundle, theta::AbstractVector)
    # theta already applied to obj.op by melitz_bundle_prepare_at_theta! -- run the KNITRO
    # solve directly against the current operator state (mirrors PsiObjectiveBundleImplicit's
    # OWN inner_loop_internal doing a moments! fill immediately before the solve; here that
    # fill already happened at the caller's own melitz_bundle_prepare_at_theta! call).
    nStatus, objSol, x = melitz_cc_inner_loop_knitro!(obj)
    if obj.mode == :implicit
        if nStatus in (0, -100, -101, -103)
            obj.x .= x
            return obj.H_save, x, nStatus
        else
            obj.x .= NaN
            MELITZ_NUMERICAL_FAILURES[] += 1
            return -1e10, x, nStatus
        end
    else
        if nStatus in (0, -100, -101, -103)
            obj.x .= x
            return objSol, x, nStatus
        else
            obj.x .= NaN
            MELITZ_NUMERICAL_FAILURES[] += 1
            return -1e10, x, nStatus
        end
    end
end

"""
    melitz_bundle_inner_loop(obj, theta) -> (val, x, nStatus)

2026-07-26 closure session (governing prompt Phase 6): a bundle-agnostic drop-in for
`CounterfactualSensitivity.inner_loop(obj, theta)` (cc_algo), for callers that want the SAME
`find_smallest`-corrected sign convention `inner_loop` applies but must not call into
`cc_algo` for a `MelitzCCBundle`. Composed ENTIRELY from the existing dispatch pair above
(`melitz_bundle_prepare_at_theta!` + `melitz_bundle_inner_solve!`) plus the SAME one-line sign
correction both `inner_loop` (cc_algo, dense bundles) and `melitz_cc_inner_loop` (this file,
`MelitzCCBundle`) already apply independently (`obj.find_smallest && (val *= -1.0)`) -- not a
new algorithm, just the missing generic entry point at the `inner_loop` level (the two
existing functions above stop one level short, at `inner_loop_internal`'s own unflipped
convention). First consumer: `nuisance_profile.jl`'s port to the production-fast backend.
"""
function melitz_bundle_inner_loop(obj, theta::AbstractVector)
    melitz_bundle_prepare_at_theta!(obj, theta)
    val, x, nStatus = melitz_bundle_inner_solve!(obj, theta)
    obj.find_smallest && (val *= -1.0)
    return val, x, nStatus
end

"""
    melitz_dense_G_from_operator(op) -> Matrix{Float64}

Diagnostic-only (Phase 8's own required escape hatch): materializes a genuine dense
`W x num_moments` `G` from the CURRENT matrix-free operator state, via `mul_Gt!` applied to
each unit basis vector (i.e. `mul_G!` with `mu=e_k`, `zeta=0`, negated) -- an O(num_moments)
number of O(W*D) sweeps, deliberately expensive, NEVER called from a production-fast hot
path (no call site in this session's production wiring does so). Exists only so an offline
cross-check script can still get a real dense `G` from a matrix-free bundle when explicitly
asked. Increments `MELITZ_DIAGNOSTIC_DENSE_SCREEN_CALLS`, never
`MELITZ_PRODUCTION_DENSE_SCREEN_CALLS`.
"""
function melitz_dense_G_from_operator(op::MelitzMomentOperator)
    K = op.layout.num_moments
    W = op.W
    G = zeros(W, K)
    mu = zeros(K)
    u = zeros(W)
    for k in 1:K
        mu[k] = 1.0
        mul_G!(u, op, 0.0, mu)
        @inbounds G[:, k] .= .-u
        mu[k] = 0.0
    end
    MELITZ_DIAGNOSTIC_DENSE_SCREEN_CALLS[] += 1
    return G
end

"""
    melitz_bundle_current_G(obj) -> Union{Nothing,AbstractMatrix}

Returns the CURRENT dense moment-matrix view for the legacy bundles (`CS.select_G_from_H`,
no recompute -- `obj.H` must already reflect the desired theta), or `nothing` for
`MelitzCCBundle` (no dense `G` exists). Used by `register_live_candidate!`
(finite_delta_outer.jl) to obtain an OPTIONAL `G_precomputed` for
`evaluate_melitz_delta_from_solution` -- `nothing` is a fully supported value there
(the matrix-free `melitz_recover_lfd_from_solution` method ignores `G_precomputed` entirely,
always using `mul_G!`/`mul_Gt!`).
"""
function melitz_bundle_current_G(obj)
    CS = CounterfactualSensitivity
    return CS.select_G_from_H(obj, obj.H)
end
melitz_bundle_current_G(obj::MelitzCCBundle) = nothing

# ============================================================================
# Phase 9: FC-to-GA / exact-point heavy-cache integration for MelitzCCBundle.
# `MelitzExactPointCache`'s heavy tier (finite_delta_outer.jl) is generalized (its
# `heavy_store` value type widened to `Any`) to hold EITHER a dense `Matrix{Float64}` (legacy
# bundles, unchanged) or a `MelitzOperatorSnapshot` (below) -- a snapshot of ONLY the
# operator's small fixed-outer-point fields (`coef`/`lambda`/`order`/`rank`/`bin`/`ell`,
# O(D^2)+O(W*D) total), never a `W x K` dense matrix. This is what lets GA reuse FC's exact
# operator state at a repeated theta with ZERO operator rebuilds (Phase 3.1's own
# `ga_operator_rebuilds=0` requirement) while staying strictly bounded in memory --
# genuinely SMALLER than the dense heavy entry it replaces (O(W*D) vs O(W*D^2)).
# ============================================================================

"""
    MelitzOperatorSnapshot

Immutable copy of `MelitzMomentOperator`'s fixed-outer-point fields (everything
`melitz_update_operator_at_theta!` writes), plus the `:implicit`-mode `H_save` scalar.
Restoring one into a LIVE operator (`melitz_heavy_restore!`) reproduces that operator's
exact state at the snapshotted theta, without rerunning `melitz_expand_theta`/
`melitz_baseline_cutoff`/`melitz_update_moment_operator!` at all.
"""
struct MelitzOperatorSnapshot
    coef::Matrix{Float64}
    lambda::Matrix{Float64}
    order::Matrix{Int}
    rank::Matrix{Int}
    bin::Matrix{UInt8}
    ell::Vector{Float64}
    fingerprint::UInt
    H_save::Float64
end

"melitz_heavy_snapshot(obj) -- generic (dense) method: a copy of obj.H."
melitz_heavy_snapshot(obj) = copy(obj.H)
"melitz_heavy_snapshot(obj::MelitzCCBundle) -- a MelitzOperatorSnapshot of obj.op's fixed-outer-point fields."
melitz_heavy_snapshot(obj::MelitzCCBundle) = MelitzOperatorSnapshot(
    copy(obj.op.coef), copy(obj.op.lambda), copy(obj.op.order), copy(obj.op.rank),
    copy(obj.op.bin), copy(obj.op.ell), obj.op.fingerprint, obj.H_save)

"melitz_heavy_restore!(obj, snap) -- generic (dense) method: obj.H .= snap."
function melitz_heavy_restore!(obj, snap::AbstractMatrix{Float64})
    obj.H .= snap
    return nothing
end
"melitz_heavy_restore!(obj::MelitzCCBundle, snap::MelitzOperatorSnapshot) -- restores op state, zero rebuild."
function melitz_heavy_restore!(obj::MelitzCCBundle, snap::MelitzOperatorSnapshot)
    op = obj.op
    copyto!(op.coef, snap.coef); copyto!(op.lambda, snap.lambda)
    copyto!(op.order, snap.order); copyto!(op.rank, snap.rank)
    copyto!(op.bin, snap.bin); copyto!(op.ell, snap.ell)
    op.fingerprint = snap.fingerprint
    obj.H_save = snap.H_save
    MELITZ_FC_TO_GA_CACHE_HITS[] += 1
    return nothing
end

"melitz_heavy_bytes(snap) -- generic (dense) method: sizeof(snap)."
melitz_heavy_bytes(snap::AbstractMatrix{Float64}) = sizeof(snap)
"melitz_heavy_bytes(snap::MelitzOperatorSnapshot) -- sum of every held array's sizeof."
melitz_heavy_bytes(snap::MelitzOperatorSnapshot) = sizeof(snap.coef) + sizeof(snap.lambda) +
    sizeof(snap.order) + sizeof(snap.rank) + sizeof(snap.bin) + sizeof(snap.ell)

"""
melitz_heavy_recompute(obj, key, ctx) -- generic (dense) method: rebuild H at theta=key via
obj.moments! (no KNITRO), matching melitz_exact_cache_get's pre-existing heavy-miss recompute.
"""
function melitz_heavy_recompute(obj, key::AbstractVector, ctx)
    CS = CounterfactualSensitivity
    H_hit = zeros(size(obj.H))
    obj.moments!(@view(H_hit[:, 1]), CS.select_G_from_H(obj, H_hit), key, obj.U, obj)
    H_hit[:, 2] .= 1.0
    return H_hit
end
"""
melitz_heavy_recompute(obj::MelitzCCBundle, key, ctx) -- rebuilds the operator at theta=key
(no KNITRO -- melitz_update_operator_at_theta! is a deterministic, cheap re-equilibration,
counted via MELITZ_OPERATOR_REBUILDS), returning a fresh snapshot.
"""
function melitz_heavy_recompute(obj::MelitzCCBundle, key::AbstractVector, ctx)
    gamma_prime_j = melitz_update_operator_at_theta!(obj.op, key, ctx)
    if obj.mode == :implicit
        obj.H_save = (gamma_prime_j - 1.0) * (-1.0)^obj.find_smallest
    end
    return melitz_heavy_snapshot(obj)
end

"""
    melitz_bundle_dense_G_at_theta(obj, theta_free) -> Matrix{Float64}

Generic (dense) method: rebuilds `G` fresh at `theta_free` via `obj.moments!` (unchanged
behavior). Used by `evaluate_melitz_delta`'s `store_G=true` path (delta_star.jl).
"""
function melitz_bundle_dense_G_at_theta(obj, theta_free::AbstractVector)
    W = size(obj.U, 1)
    K = zeros(W)
    G = zeros(W, obj.d)
    obj.moments!(K, G, theta_free, obj.U, obj)
    return G
end
"""
melitz_bundle_dense_G_at_theta(obj::MelitzCCBundle, theta_free) -- diagnostic-only dense
materialization from the operator (assumed ALREADY at theta_free, e.g. immediately after
melitz_recover_lfd(obj, theta_free) -- never re-updates the operator itself).
"""
melitz_bundle_dense_G_at_theta(obj::MelitzCCBundle, theta_free::AbstractVector) = melitz_dense_G_from_operator(obj.op)

"""
    fstar_equal_weight_moments(theta_free, ctx, obj::MelitzCCBundle) -> Vector{Float64}

Matrix-free method for fstar_direct.jl's `fstar_equal_weight_moments` (the direct F*
feasibility solve's own moment-matching objective, `m_Fstar(theta) = vec(mean(G(theta),
dims=1))`). Genuinely matrix-free, not merely a dense-G-then-average shortcut: `mean(G,
dims=1) = G' * (ones(W)/W)`, exactly what `mul_Gt!` already computes for any weight vector
-- one `O(W*D)` call, no `W x K` materialization at all (cheaper than the generic dense
method this overrides, not just an equally-expensive rerouting). Updates the operator at
`theta_free` first (this function, unlike `melitz_bundle_dense_G_at_theta`, is called at
ARBITRARY search points by an `Optim.jl` driver, not immediately after an inner solve that
already moved the operator there).
"""
function fstar_equal_weight_moments(theta_free::AbstractVector{<:Real}, ctx, obj::MelitzCCBundle)
    melitz_update_operator_at_theta!(obj.op, theta_free, ctx)
    W = obj.op.W
    v = fill(1.0 / W, W)
    g = zeros(obj.d)
    mul_Gt!(g, obj.op, v)
    return g
end
