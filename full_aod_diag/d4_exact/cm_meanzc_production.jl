# ============================================================================
# CM+mean(+ZC) production bundle: inner solve, structured Hessian, gradient.
# Companion to cm_meanzc_moments.jl. Reuses, UNCHANGED:
#   - Architecture C's CMBinHessCtx / hessian_cm_structured! / archC_hess_cb_builder
#     (cm_hessian_architectures.jl) -- only the E-block width (NCORE) widens;
#     the Hessian math needs zero new code (see build_cm_meanzc_bin_ctx below).
#   - inner_loop_internal_archgeneric / inner_loop_KNITRO_archgeneric
#     (cm_hessian_architectures.jl) -- called with a WIDER theta_ext = vcat(theta_econ, nu),
#     which they forward verbatim to obj.moments! (they do not inspect theta's length).
#   - composite_gradient_at_fast (lfix_incremental.jl/composite_gradient.jl,
#     via lfix_cm_aware.jl's with_q0 pattern) -- the (g,A_od) outer gradient is
#     computed EXACTLY as in the CM-only/CM-only path, at nu held fixed, by
#     folding the mean/pair fixed contribution into q0 once per outer
#     evaluation (see meanzc_fixed_contribution below) -- no per-coordinate
#     probe ever re-touches the mean/pair/CM blocks.
# CMExpectedSolveFailure (cm_production_bundle.jl) is reused for the same
# expected-failure signal, not redefined.
# ============================================================================

using LinearAlgebra: BLAS, dot

"""
    build_cm_meanzc_bin_ctx(ctx, aug) -> CMBinHessCtx

Architecture-C precomputation for a `build_cm_meanzc_augmented_obj` result.
Identical to `build_cm_bin_ctx` (cm_hessian_architectures.jl) except `NCORE`
is widened to `aug.ncore_econ + aug.n_mean + aug.n_pair` (the mean/pair
columns join the dense "economic" BLAS block, per cm_meanzc_moments.jl's
column layout) -- `CMBinHessCtx`, `hessian_cm_structured!`, and
`archC_hess_cb_builder` themselves are reused completely unmodified.
"""
function build_cm_meanzc_bin_ctx(ctx, aug)
    L = aug.L; D = ctx.D; origins = aug.origins; nO = length(origins)
    refIndex1 = aug.refIndex1; z = aug.z
    NCORE_ext = aug.ncore_econ + aug.n_mean + aug.n_pair
    ncm = aug.ncm
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = compute_bin_indices(ctx.U, z)
    W = size(ctx.U, 1)
    L1 = L + 1
    return CMBinHessCtx(L, D, nO, origins, refIndex1, z, Bidx, NCORE_ext, ncm, aug.contrasts, R,
        zeros(D, D, L1, L1), zeros(D, NCORE_ext, L1), zeros(D, D, L, L), zeros(D, NCORE_ext, L),
        Matrix{Float64}(undef, W, NCORE_ext), Matrix{Float64}(undef, NCORE_ext + ncm, NCORE_ext + ncm))
end

"""
    build_cm_meanzc_production_context(ctx, CS; L, cm_extension, contrasts=:orthonormal,
        meanzc_basis=:direct, probs=nothing) -> (ctx_cm, aug, cctx)

Analog of `build_cm_production_context` (cm_production_bundle.jl) for the
extended arms. `ctx_cm.obj` is `aug.obj_cm` (theta-independent Bidx/z/Zraw/
Zpairraw precomputed ONCE, reused across every subsequent inner solve at this
context, exactly matching the CM-only production context's own performance
contract).
"""
function build_cm_meanzc_production_context(ctx, CS; L::Int, cm_extension::Symbol,
                                             contrasts::Symbol = :orthonormal, meanzc_basis::Symbol = :direct,
                                             probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = cm_extension,
        contrasts = contrasts, meanzc_basis = meanzc_basis, probs = probs)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    bins = cm_bin_indices_for(ctx, aug)   # lfix_cm_aware.jl -- Unsigned-typed, for the CM fixed-contribution lookup
    return (ctx_cm = ctx_cm, aug = aug, cctx = cctx, bins = bins)
end

"""
    archC_meanzc_base_state(x_free0, ν, ctx_cm, cctx) -> BaseDualState

Architecture-C inner dual solve at outer point `(x_free0, ν)`. `ν` is passed
explicitly and rides through as `theta_ext[end]` (see cm_meanzc_moments.jl
header) -- no shared/cached state. Mirrors `archC_base_state`
(cm_production_bundle.jl) exactly otherwise.
"""
function archC_meanzc_base_state(x_free0::AbstractVector, ν::Float64, ctx_cm, cctx::CMBinHessCtx)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    θ_ext0 = vcat(θ_econ0, ν)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_ext0;
        hess_cb_builder = _obj -> archC_hess_cb_builder(cctx))
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archC_meanzc_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0, ν=$ν)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    archC_meanzc_verified_state(x_free0, ν, ctx_cm, cctx) -> (base::BaseDualState, verify::NamedTuple)

AUD-04-style verified analog of `archC_meanzc_base_state`, mirroring
`archC_verified_state` (cm_production_bundle.jl): same inner solve, PLUS the
independent residual/gap diagnostics `classify_inner_result`/
`is_verified_success` (oracle.jl) need, computed via the same explicit-recompute
pattern (never trusts KNITRO's last FG callback alone).
"""
function archC_meanzc_verified_state(x_free0::AbstractVector, ν::Float64, ctx_cm, cctx::CMBinHessCtx)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    θ_ext0 = vcat(θ_econ0, ν)
    K, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_ext0;
        hess_cb_builder = _obj -> archC_hess_cb_builder(cctx))
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archC_meanzc_verified_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0, ν=$ν)"))

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = size(obj.U, 1)
    G = CS.select_G_from_H(obj, obj.H)

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = primal_divergence(m_weights)

    mean_m_resid = abs(sum(m_weights) / W - 1.0)
    nkkt = min(length(λstar), size(G, 2))
    max_abs_moment_kkt_resid = kkt_residual_blas(G, m_weights, nkkt, W)

    base = BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, m_weights, nStatus)
    verify = (inner_status = nStatus, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              m_mean = sum(m_weights) / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
    return base, verify
end

"cm_meanzc_production_value(x_free0, ν, pcx) -> (K, base). Inner solve only, no gradient."
function cm_meanzc_production_value(x_free0::AbstractVector, ν::Float64, pcx)
    base = archC_meanzc_base_state(x_free0, ν, pcx.ctx_cm, pcx.cctx)
    K = pcx.ctx_cm.obj.H_save
    return K, base
end

"cm_meanzc_production_value_verified(x_free0, ν, pcx) -> (K, base, verify). AUD-04-gated analog."
function cm_meanzc_production_value_verified(x_free0::AbstractVector, ν::Float64, pcx)
    base, verify = archC_meanzc_verified_state(x_free0, ν, pcx.ctx_cm, pcx.cctx)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end

# ----------------------------------------------------------------------------
# Outer (g, A_od) gradient at fixed ν, exact structural analog of
# lfix_cm_aware.jl's build_lfix_base_cache_cm / composite_gradient_at_fast_cm,
# generalized to also fold in the mean/pair fixed contribution. ν is passed
# explicitly (never read from a stored field) -- this is the ONLY place ν
# needs to be known outside the one inner solve above, and it is needed only
# ONCE per outer evaluation (to build q0), not per coordinate probe.
# ----------------------------------------------------------------------------

"""
    meanzc_fixed_contribution(base::BaseDualState, aug, ν::Float64) -> Vector{Float64}

`out[s] = λ_mean*'·(Z_s - ν) + λ_pair*'·(Zpair_s - ν²)` for every draw `s`
(mean-block term only if `aug.n_pair == 0`). O(W*(D+n_pair)) via BLAS
matrix-vector products (the mean/pair columns are raw dense columns, not
bin-indexed step functions, so no suffix-sum lookup is needed here, unlike
the CM-grid block).
"""
function meanzc_fixed_contribution(base::BaseDualState, aug, ν::Float64)
    ncore_econ = aug.ncore_econ; n_mean = aug.n_mean; n_pair = aug.n_pair
    @assert length(base.λstar) >= ncore_econ - 1 + n_mean + n_pair "base.λstar too short for aug's (ncore_econ,n_mean,n_pair) -- was base solved against aug.obj_cm?"
    λ_mean = @view base.λstar[ncore_econ:ncore_econ+n_mean-1]
    out = aug.Zraw * λ_mean
    out .-= ν * sum(λ_mean)
    if n_pair > 0
        λ_pair = @view base.λstar[ncore_econ+n_mean:ncore_econ+n_mean+n_pair-1]
        out .+= aug.Zpairraw * λ_pair
        out .-= ν^2 * sum(λ_pair)
    end
    return out
end

"""
    cm_fixed_contribution_meanzc_layout(base, ctx, aug, bins) -> Vector{Float64}

Identical math to `cm_fixed_contribution` (lfix_cm_aware.jl), re-sliced at the
CM-grid block's actual position under the
`[economic | mean | pair | CM-grid | gravity]` layout (starts at
`ncore_econ + n_mean + n_pair`, not `ncore_econ` as in the plain CM `aug`).
"""
function cm_fixed_contribution_meanzc_layout(base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned})
    ncore_econ = aug.ncore_econ; n_mean = aug.n_mean; n_pair = aug.n_pair; ncm = aug.ncm; L = aug.L
    nO = length(aug.origins)
    cm_start = ncore_econ + n_mean + n_pair
    @assert length(base.λstar) >= cm_start - 1 + ncm "base.λstar too short for aug's (ncore_econ,n_mean,n_pair,ncm) -- was base solved against aug.obj_cm?"
    λ_cm = base.λstar[cm_start:cm_start-1+ncm]
    λmat_stored = reshape(λ_cm, nO, L)
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    λmat_block = apply_contrast(λmat_stored, R)
    P = suffix_sums(λmat_block)
    out = Vector{Float64}(undef, size(bins, 1))
    cumulative_forward_contribution!(out, bins, aug.refIndex1, aug.origins, P)
    return out
end

"""
    build_lfix_base_cache_cm_meanzc(x_free0, ctx_cm, base, ctx, aug, bins, ν; validate_dense=false) -> LFixBaseCache

CM+mean(+ZC)-aware analog of `build_lfix_base_cache_cm` (lfix_cm_aware.jl).
`ν` is the current outer evaluation's ν, passed explicitly.
"""
function build_lfix_base_cache_cm_meanzc(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                          ctx, aug, bins::AbstractMatrix{<:Unsigned}, ν::Float64;
                                          validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_fixed_contribution_meanzc_layout(base, ctx, aug, bins)
    meanzc_contrib0 = meanzc_fixed_contribution(base, aug, ν)
    return with_q0(cache0, cache0.q0 .- cm_contrib0 .- meanzc_contrib0)
end

"""
    cm_meanzc_production_gradient(x_free0, ν, pcx, ctx, pe; base=nothing, verify=nothing, kwargs...) -> (g_ext, meta)

One-call entry point for the extended arms' full outer gradient: the (g,A_od)
block via the UNCHANGED `composite_gradient_at_fast` (ν held fixed, folded
into q0 once, matching `composite_gradient_at_fast_cm`'s own contract), PLUS
the analytic `∂Delta_dual/∂η_ν` appended as the LAST component. `g_ext` has
length `n_free(pe's map) + 1`. If `verify` (from `cm_meanzc_production_value_verified`)
is not supplied, one extra verified inner solve is performed to get `m_mean`.
"""
function cm_meanzc_production_gradient(x_free0::AbstractVector, ν::Float64, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, verify = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archC_meanzc_verified_state(x_free0, ν, pcx.ctx_cm, pcx.cctx)
    end
    cache = build_lfix_base_cache_cm_meanzc(x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins, ν)
    g_econ, meta = composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
    d_eta = d_delta_dual_d_eta_nu(base.λstar, pcx.aug, ν; mean_m = verify.m_mean)
    return vcat(g_econ, d_eta), meta
end

# ----------------------------------------------------------------------------
# Validation ground truth for the (g, A_od) gradient block, meanzc-aware
# analog of fixed_dual_L / full_rebuild_gradient_fallback (composite_gradient_fast.jl,
# three_way_derivatives.jl). composite_gradient_at_fast is itself an adaptive-
# bandwidth secant method around a possibly-nonsmooth (winner-switching)
# objective -- a naive fixed-h central difference is NOT a valid ground truth
# for it (confirmed live: plain CM's own already-trusted analytic gradient
# disagrees with a naive h=1e-5 probe by up to 0.14, but matches
# full_rebuild_gradient_fallback's h=0.01 fixed-dual full-rebuild reference to
# cosine similarity 0.9999 / max abs diff 1.5e-3). These functions give the
# SAME trusted reference for the meanzc-augmented objective.
# ----------------------------------------------------------------------------

"""
    fixed_dual_L_meanzc(x_free, ν, ctx_cm, base::BaseDualState) -> Float64

Meanzc-aware analog of `fixed_dual_L` (three_way_derivatives.jl): the fixed-
dual (NOT reoptimized) divergence value at a perturbed `(x_free, ν)`, holding
`base.ζstar`/`base.λstar` fixed at their converged values. Rebuilds the FULL
moment matrix via `obj.moments!` (theta_ext = vcat(theta_econ, ν)) every call
-- slower than the incremental machinery, always correct, matching
`fixed_dual_L`'s own "no shortcuts" contract.
"""
function fixed_dual_L_meanzc(x_free::AbstractVector, ν::Float64, ctx_cm, base::BaseDualState)
    obj = ctx_cm.obj
    θ_econ = CS.reconstruct_full(x_free, ctx_cm.m)
    θ_ext = vcat(θ_econ, ν)
    W = size(obj.U, 1); d = obj.d
    K = zeros(eltype(θ_ext), W); G = zeros(eltype(θ_ext), W, d)
    obj.moments!(K, G, θ_ext, obj.U, obj)
    oci = obj.outer_constr_index
    q = [-base.ζstar - dot(base.λstar, @view(G[s, 1:oci-1])) for s in 1:W]
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return -(sum(Psi_q) / W + base.ζstar)
end

"""
    full_rebuild_gradient_fallback_meanzc(x_free0, ν, ctx_cm, pe, base; h=0.01) -> (g, meta)

Meanzc-aware analog of `full_rebuild_gradient_fallback`: full-rebuild fixed-
dual central FD over the (g, A_od) block ONLY (ν held fixed throughout, matching
`composite_gradient_at_fast`'s own contract for this block), via
`fixed_dual_L_meanzc`. Use this, not a naive ad hoc FD probe, as the reference
for validating `cm_meanzc_production_gradient`'s (g,A_od) block.
"""
function full_rebuild_gradient_fallback_meanzc(x_free0::AbstractVector, ν::Float64, ctx_cm, pe, base::BaseDualState; h::Float64 = 0.01)
    D = ctx_cm.D; D2 = D^2
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))
    g = zeros(D2)
    for k in 1:D2
        wp = copy(w0); wp[k] += h
        wm = copy(w0); wm[k] -= h
        Lp = fixed_dual_L_meanzc(x_free_from_w(wp, pe), ν, ctx_cm, base)
        Lm = fixed_dual_L_meanzc(x_free_from_w(wm, pe), ν, ctx_cm, base)
        g[k] = (Lp - Lm) / (2h)
    end
    return g, (base = base, w0 = w0, h_used = fill(h, D2), method = :full_rebuild_fallback_meanzc)
end
