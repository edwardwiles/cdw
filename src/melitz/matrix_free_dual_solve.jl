# Matrix-free Melitz inner CC dual solve (2026-07-26 continuation, PART 3). Validates
# `mul_G!`/`mul_Gt!`/`melitz_full_weighted_gram!` (moment_operator.jl) through a REAL KNITRO
# inner solve -- not merely against a dense reference at an isolated point, but by actually
# driving KNITRO's own optimizer to convergence using ONLY the matrix-free operator for the
# objective/gradient/Hessian, then comparing the converged dual, objective value, and
# recovered LFD/moment-residual diagnostics against the EXISTING dense
# `PsiObjectiveBundleDelta`/`inner_loop`/`melitz_recover_lfd` path on the IDENTICAL
# (p, eq, cf, z_draws) fixture.
#
# DELIBERATE DESIGN CHOICE: this file does NOT touch `cc_algo/PsiObjectiveBundle.jl` or
# `cc_algo/inner_loop_functions.jl` at all -- no shared type is subtyped, no shared generic
# function is extended, no shared file is edited. `MelitzMatrixFreeDualBundle` is a
# freestanding struct with its OWN functor and its OWN small KNITRO driver
# (`melitz_matrix_free_inner_solve`), mirroring `PsiObjectiveBundleDelta`'s functor and
# `inner_loop_KNITRO`'s own KNITRO API call sequence line-for-line, but never calling into
# or modifying that code. This keeps the blast radius at exactly zero for the Ricardian
# model (nothing in `cc_algo` changes) while still being a genuinely REAL KNITRO solve --
# same KNITRO.jl API, same options file, same callback mechanism, same divergence conjugate
# (`Psi!`/`dPsi!`/`ddPsi!`, copied here VERBATIM from `cc_algo/Psi.jl` rather than imported,
# so this file has no hard load-order dependency on `cc_algo` at all beyond needing the
# `KNITRO` package itself).
#
# SCOPE: this validates the INNER, fixed-outer-point dual solve ONLY (mirroring
# `PsiObjectiveBundleDelta`'s own documented scope -- "exclusively for FIXED-theta inner CC
# dual solves"). It does NOT implement the outer theta-gradient branch
# (`calculate_jac_θ!`/`ift!`/`jac_h`) -- a full matrix-free OUTER search is a separate,
# larger undertaking, not attempted here.

using KNITRO

"""
    MelitzMatrixFreeDualBundle

Freestanding KNITRO-callable bundle for the Melitz inner CC dual program, using
`MelitzMomentOperator` (never a materialized `G`) for every objective/gradient/Hessian
evaluation. `op` must already be updated (`melitz_update_moment_operator!`) at the outer
point being solved -- this bundle does NOT rebuild it (mirrors `PsiObjectiveBundleDelta`'s
own `obj.moments!` being called ONCE per `inner_loop_internal`, before KNITRO ever starts
iterating, not on every callback).

`x = [zeta; mu]` (length `d+1`, `d = op.layout.num_moments`), matching
`PsiObjectiveBundleDelta`'s own `(ζ, λ)` convention exactly (`ζ=x[1]`, `λ=x[2:end]`).
"""
mutable struct MelitzMatrixFreeDualBundle
    op::MelitzMomentOperator
    M::Int
    d::Int
    outer_constr_index::Int
    lower_limit::Float64
    use_cached_x::Bool
    x::Vector{Float64}
    arg0::Vector{Float64}   # u = -zeta - dot(G,mu)
    arg1::Vector{Float64}   # Psi(u) or dPsi(u), reused (matches PsiObjectiveBundleDelta's own arg1 reuse)
    arg2::Vector{Float64}   # ddPsi(u) = S, the curvature weight vector
    Hfull::Matrix{Float64}  # (d+1) x (d+1) scratch for melitz_full_weighted_gram!
end

"""
    build_melitz_matrix_free_dual_bundle(op; lower_limit=-KNITRO.KN_INFINITY) -> MelitzMatrixFreeDualBundle

`op` must already be updated at the outer point to be solved
(`melitz_update_moment_operator!`).
"""
function build_melitz_matrix_free_dual_bundle(op::MelitzMomentOperator;
                                               lower_limit::Float64=-KNITRO.KN_INFINITY)
    W = op.W
    d = op.layout.num_moments
    return MelitzMatrixFreeDualBundle(op, W, d, d + 1, lower_limit, false, fill(NaN, d + 1),
        zeros(W), zeros(W), zeros(W), zeros(d + 1, d + 1))
end

# Divergence conjugate (hybrid exponential/quadratic tilting) -- copied VERBATIM from
# cc_algo/Psi.jl's Psi!/dPsi!/ddPsi! (byte-for-byte identical arithmetic), under
# Melitz-local names to avoid any load-order coupling to cc_algo.
function _mf_Psi!(arg1::AbstractVector{Float64}, arg0::AbstractVector{Float64})
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

function _mf_dPsi!(arg1::AbstractVector{Float64}, arg0::AbstractVector{Float64})
    @inbounds for i in 1:length(arg0)
        arg1[i] = arg0[i] <= 1.0 ? exp(arg0[i]) : exp(1) * arg0[i]
    end
    return arg1
end

function _mf_ddPsi!(arg1::AbstractVector{Float64}, arg0::AbstractVector{Float64})
    @inbounds for i in 1:length(arg0)
        arg1[i] = arg0[i] <= 1.0 ? exp(arg0[i]) : exp(1)
    end
    return arg1
end

"""
    (Q::MelitzMatrixFreeDualBundle)(x, g=Float64[]; h=Float64[]) -> f::Float64

Objective/gradient/Hessian functor, matching `PsiObjectiveBundleDelta`'s own functor
(`cc_algo/PsiObjectiveBundle.jl`) EXACTLY in mathematical content -- `f = sum(Psi(u))/M +
zeta`, `g[1] = 1 - sum(dPsi(u))/M`, `g[2:end] = -(1/M)*G'*dPsi(u)`, Hessian `=
(1/M)*[ones(W) G]' * Diag(ddPsi(u)) * [ones(W) G]` -- but every `G`-touching step goes
through `mul_G!`/`mul_Gt!`/`melitz_full_weighted_gram!` instead of `BLAS.gemv!`/`gemm!` on a
dense `H`. Matches the EXACT calling convention `cc_algo/inner_loop_functions.jl`'s
`callbackEvalFG_inner!`/`callbackEvalH_inner!` use (`obj(x, evalResult.objGrad)` and
`obj(x, h=evalResult.hess)` -- no `theta`/`constr`/`jac` arguments, since those belong only
to the OUTER theta-gradient branch this bundle does not implement).
"""
function (Q::MelitzMatrixFreeDualBundle)(x::AbstractVector{Float64}, g::AbstractVector{Float64}=Float64[];
                                          h::AbstractVector{Float64}=Float64[])
    zeta = x[1]
    mu = @view x[2:end]

    mul_G!(Q.arg0, Q.op, zeta, mu)
    _mf_Psi!(Q.arg1, Q.arg0)
    f = sum(Q.arg1) / Q.M + zeta

    if length(g) > 0 || length(h) > 0
        _mf_dPsi!(Q.arg1, Q.arg0)
    end

    if length(g) > 0
        g[1] = 1.0 - sum(Q.arg1) / Q.M
        gmu = @view g[2:end]
        mul_Gt!(gmu, Q.op, Q.arg1)
        gmu .*= -1.0 / Q.M
    end

    if length(h) > 0
        _mf_ddPsi!(Q.arg2, Q.arg0)
        melitz_full_weighted_gram!(Q.Hfull, Q.op, Q.arg2)
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

    if f <= Q.lower_limit
        return -KNITRO.KN_INFINITY
    else
        return f
    end
end

"""
    melitz_matrix_free_inner_solve(bundle, inner_loop_opt) -> (nStatus, objSol, x)

Standalone KNITRO driver, mirroring `cc_algo/inner_loop_functions.jl`'s own
`inner_loop_KNITRO` call sequence (same KNITRO API calls, same options-file loading, same
Hessian-callback registration gated on `hessopt`) but built directly around
`MelitzMatrixFreeDualBundle` -- no `ObjectiveBundle` subtyping, no shared dispatch. Free
(unbounded) variables throughout, matching `PsiObjectiveBundleDelta`'s own
`inner_loop_lower_bounds` for the Melitz case (`inequality_index` is always empty in this
codebase's actual Melitz moment system -- every moment is an equality).
"""
function melitz_matrix_free_inner_solve(bundle::MelitzMatrixFreeDualBundle, inner_loop_opt::String)
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
    KNITRO.KN_load_param_file(kc, inner_loop_opt)
    if melitz_kn_get_int_param(kc, "hessopt") == 1
        KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, cbEvalH!)
    end

    KNITRO.KN_solve(kc)
    nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    if nStatus in (0, -100, -101, -103)
        bundle.x .= x
    end
    return nStatus, objSol, x
end

"""
    melitz_matrix_free_moment_residuals(bundle, x) -> (weights, moment_residuals, normalization_residual)

Recovers the LFD weights and per-moment residuals from a converged dual `x`, matrix-free
(mirrors `delta_star.jl`'s own `melitz_recover_lfd_from_solution` but via `mul_G!`/`mul_Gt!`
instead of a dense `G`). `moment_residuals[k] = sum_w weights[w]*G[w,k]` for every moment
column `k` (trade cells then the focal link), via ONE `mul_Gt!` call on the (normalized)
LFD weights.
"""
function melitz_matrix_free_moment_residuals(bundle::MelitzMatrixFreeDualBundle, x::AbstractVector{Float64})
    zeta = x[1]
    mu = @view x[2:end]
    u = zeros(bundle.M)
    mul_G!(u, bundle.op, zeta, mu)
    lfd = zeros(bundle.M)
    _mf_dPsi!(lfd, u)
    s = sum(lfd)
    weights = lfd ./ s
    moment_residuals = zeros(bundle.d)
    mul_Gt!(moment_residuals, bundle.op, weights)
    return weights, moment_residuals, s / bundle.M - 1
end
