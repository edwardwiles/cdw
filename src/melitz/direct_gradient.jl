# Continuation session (2026-07-23->2026-07-24, "make the optimized architecture scalable
# in memory and D") Section 4: the DIRECT fixed-dual gradient-VECTOR backend.
#
# -- Why this exists on top of `:B_argument_localized_serial`/`_parallel` --
#
# The argument-localized backend (`argument_localized_gradient.jl`) already avoids building
# any full `(W, K)` displaced-moment matrix, but it is still wired in as a `moments_jacobian!`
# -- it WRITES into `select_jac_g_from_jac_h(obj, obj.jac_h)`, a VIEW of the bundle's own
# PRE-ALLOCATED `jac_h::Array{Float64,3}` tensor (size `N x (d+2) x l`). Even though only
# `O(D)` columns per coordinate ever get a nonzero value, the view itself spans the FULL
# `(N, d, l)` shape, so `fill!(G_jac, 0.0)` -- required so every untouched column reads back
# exactly `0.0` for the downstream `calculate_jac_θ!`/envelope-theorem contraction
# (`cc_algo/PsiObjectiveBundle.jl`'s `PsiObjectiveBundleImplicit` functor, theta-branch) --
# is an `O(W*K*n)` memset: ~190.7GB of zero-fill traffic for a SINGLE gradient call at
# D=20/W=80,000 (continuation3's own Section 2 memory audit). The bundle must also still
# ALLOCATE the full `jac_h` tensor at construction (`~206GB` at that scale) purely to have
# somewhere for that view to point, even though only a tiny fraction is ever meaningfully
# written.
#
# This backend eliminates BOTH costs by never going through `jac_h`/`calculate_jac_θ!` at
# all. It reuses the SAME two facts the argument-localized backend already established:
#
#   1. `melitz_compact_columns_map(ctx)` (Gate 1, validated) already lists, per free
#      coordinate, the SMALL set of physical trade cells (`O(D)`, not `O(D^2)`) whose moment
#      column can possibly change under a perturbation of that coordinate.
#   2. The bundle's own `obj.H` already holds the moment matrix `G` at the CURRENT
#      (unperturbed) theta -- `obj.H[:, 2+k]` is column `k` of `G` -- so the BASE value of
#      any touched column is available for free, with no separate `Gbase` matrix needed.
#
# What is fundamentally different from `:B_argument_localized_*`: this backend computes the
# FINAL, length-`n_theta` gradient of the divergence constraint row directly, following the
# "direct fixed-dual gradient-vector" recipe (main prompt Section 4):
#
#   For each free coordinate `r`, and using the FIXED dual `x = (ζ, λ)` from the just-solved
#   inner CC problem (never re-solved -- this is the same fixed-dual/envelope-theorem
#   approximation `PsiObjectiveBundleImplicit`'s own analytic contraction already relies on):
#
#     u_base[s]  = -ζ - dot(G_base[s, :], λ)                          (= obj's own `arg0`)
#     u_plus[s]  = u_base[s] - sum_{k touched by r} λ[k]*(G_plus[s,k]  - G_base[s,k])
#     u_minus[s] = u_base[s] - sum_{k touched by r} λ[k]*(G_minus[s,k] - G_base[s,k])
#     L_plus     = sum_s Psi(u_plus[s])  / W
#     L_minus    = sum_s Psi(u_minus[s]) / W
#     grad[r]    = -1e10 * (L_plus - L_minus) / (2h)
#
#   (the additive `+ζ` term of the raw functor's own `f = sum(arg1)/M + ζ` cancels exactly
#   in the plus/minus difference, since `ζ` does not depend on `theta`, so it is omitted).
#
# This is a genuinely INDEPENDENT reconstruction from the existing analytic jac_h-contraction
# path -- it never calls `Psi!`'s derivative `dPsi!` or contracts through `jac_h`, only the
# forward `Psi!` itself, evaluated at two DISPLACED fixed-dual arguments -- so comparing its
# output against `:B_argument_localized_parallel`'s (Section 6 validation) is a genuine
# cross-check of the whole envelope-theorem gradient construction, not merely two code paths
# computing the identical formula.
#
# Sign/scale convention: chosen so `grad[r] == d(1e10*Delta(theta))/dtheta_r`, EXACTLY the
# quantity `cb_G!`'s existing `local_jac` (via `obj(x, dummy_g, theta; jac=local_jac)`)
# already produces via the analytic path -- see `finite_delta_outer.jl`'s own `constr[1] =
# -f*1e10` / `Delta_theta = local_c[1]/1e10` convention (Section 18's sign fix). This lets
# the new backend be a drop-in replacement inside `cb_G!` for the SAME KNITRO constraint row,
# with no change to any surrounding scaling/sign logic.
#
# Memory: the ONLY per-call allocations are `O(W)` (arg0/u_plus/u_minus, shared for the
# serial variant, one copy per thread for the parallel variant) and `O(W*maxcols)`
# (Gp/Gm probe buffers, `maxcols = O(D)`, identical sizing to the argument-localized
# backend's own scratch) -- NEVER `O(W*K)` or `O(W*K*n)`. No `jac_h` tensor is read, written,
# or allocated by this backend at all (the bundle it operates on is constructed with
# `needs_outer_moment_jacobian=false`).

using LinearAlgebra: BLAS

"""
    _base_arg0!(arg0_base, obj, x) -> arg0_base

Fills `arg0_base[s] = -ζ - dot(G_base[s,:], λ)` from the bundle's own CURRENT `obj.H`
(columns `2:1+outer_constr_index`, i.e. `[ones(M) G]`) and dual `x = (ζ, λ)` -- the exact
same BLAS contraction `PsiObjectiveBundleImplicit`'s own functor performs internally
(`cc_algo/PsiObjectiveBundle.jl`, non-gradient branch), reused here rather than re-derived,
so this backend's "base point" is provably the SAME base point the reference path uses.
"""
function _base_arg0!(arg0_base::AbstractVector{Float64}, obj, x::AbstractVector{Float64})
    outer_constr_index = obj.outer_constr_index
    BLAS.gemv!('N', 1.0, @view(obj.H[:, 2:1+outer_constr_index]), -x, 0.0, arg0_base)
    return arg0_base
end

"""
    _direct_coordinate_grad(cc, theta_p, theta_m, ctx, obj, lambda, arg0_base, h,
                             Gp, Gm, linkp, linkm, profit, u_plus, u_minus, psi_buf) -> Float64

One free coordinate's contribution: builds `u_plus`/`u_minus` from `arg0_base` plus ONLY the
touched-column contributions (direct trade cells + focal link if applicable), evaluates the
fixed-dual scalar objective `Psi!` at each, and returns the central-difference gradient
`-1e10 * (L_plus - L_minus) / (2h)`. All buffers are caller-owned scratch (length `W` or
`(W, maxcols)`), reused across coordinates/threads -- no allocation inside this function.
"""
function _direct_coordinate_grad(cc::MelitzCompactColumns, theta_p::AbstractVector{Float64},
                                  theta_m::AbstractVector{Float64}, ctx, obj,
                                  lambda::AbstractVector{Float64}, arg0_base::AbstractVector{Float64},
                                  h::Real, Gp::AbstractMatrix{Float64}, Gm::AbstractMatrix{Float64},
                                  linkp::AbstractVector{Float64}, linkm::AbstractVector{Float64},
                                  profit::AbstractVector{Float64}, u_plus::AbstractVector{Float64},
                                  u_minus::AbstractVector{Float64}, psi_buf::AbstractVector{Float64})
    W = length(arg0_base)
    layout = ctx.moment_layout
    ncols = length(cc.direct_cols)

    ncols > 0 && _fill_compact_direct_columns!(Gp, theta_p, ctx, obj, cc.direct_cells, ncols)
    ncols > 0 && _fill_compact_direct_columns!(Gm, theta_m, ctx, obj, cc.direct_cells, ncols)
    if cc.touches_link
        _fill_compact_link!(linkp, profit, theta_p, ctx, obj)
        _fill_compact_link!(linkm, profit, theta_m, ctx, obj)
    end

    copyto!(u_plus, arg0_base)
    copyto!(u_minus, arg0_base)
    @inbounds for idx in 1:ncols
        gcol = cc.direct_cols[idx]
        lam_k = lambda[gcol]
        Gbase_col_offset = 2 + gcol   # obj.H layout: [K, 1, G...] -> column gcol of G is H[:, 2+gcol]
        for w in 1:W
            gbase = obj.H[w, Gbase_col_offset]
            u_plus[w]  -= lam_k * (Gp[w, idx] - gbase)
            u_minus[w] -= lam_k * (Gm[w, idx] - gbase)
        end
    end
    if cc.touches_link
        lam_link = lambda[layout.focal_link_index]
        link_col_offset = 2 + layout.focal_link_index
        @inbounds for w in 1:W
            gbase = obj.H[w, link_col_offset]
            u_plus[w]  -= lam_link * (linkp[w] - gbase)
            u_minus[w] -= lam_link * (linkm[w] - gbase)
        end
    end

    obj.Psi!(psi_buf, u_plus)
    L_plus = sum(psi_buf) / W
    obj.Psi!(psi_buf, u_minus)
    L_minus = sum(psi_buf) / W

    return -1e10 * (L_plus - L_minus) / (2h)
end

"""
    make_melitz_gradient_delta_direct_serial(h) -> Function

Backend `:B_direct_argument_serial` (main prompt Section 4). Returns a closure
`(g, theta, ctx, obj, x) -> nothing` filling `g[r] = d(1e10*Delta(theta))/dtheta_r` for
every free coordinate `r`, directly, with NO `jac_h`/`moments_jacobian!` involvement at
all. Persistent scratch (`O(W)` vectors, one `O(W*maxcols)` probe-buffer pair) is closed
over and reused across calls, rebuilt only on a `(W, maxcols)` shape change (mirrors
`argument_localized_gradient.jl`'s own caching discipline).
"""
function make_melitz_gradient_delta_direct_serial(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    arg0_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    Gp_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gm_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    linkp_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    linkm_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    profit_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    uplus_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    uminus_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    psi_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    thetap_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    thetam_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)

    function melitz_gradient_delta_direct_serial!(g::AbstractVector{Float64}, theta::AbstractVector{Float64},
                                                    ctx, obj, x::AbstractVector{Float64})
        n = length(theta)
        W = size(obj.U, 1)

        if compact_cache[] === nothing || ctx_cache[] !== ctx
            compact_cache[] = melitz_compact_columns_map(ctx)
            ctx_cache[] = ctx
        end
        compact = compact_cache[]
        maxcols = maximum(length(c.direct_cols) for c in compact)
        if arg0_buf[] === nothing || length(arg0_buf[]) != W || size(Gp_buf[]) != (W, maxcols)
            arg0_buf[] = zeros(Float64, W)
            Gp_buf[] = zeros(Float64, W, maxcols)
            Gm_buf[] = zeros(Float64, W, maxcols)
            linkp_buf[] = zeros(Float64, W)
            linkm_buf[] = zeros(Float64, W)
            profit_buf[] = zeros(Float64, W)
            uplus_buf[] = zeros(Float64, W)
            uminus_buf[] = zeros(Float64, W)
            psi_buf[] = zeros(Float64, W)
        end
        if thetap_buf[] === nothing || length(thetap_buf[]) != n
            thetap_buf[] = zeros(Float64, n)
            thetam_buf[] = zeros(Float64, n)
        end
        arg0_base = arg0_buf[]
        _base_arg0!(arg0_base, obj, x)
        lambda = @view x[2:end]

        theta_p = thetap_buf[]
        theta_m = thetam_buf[]
        for r in 1:n
            cc = compact[r]
            copyto!(theta_p, theta); theta_p[r] += h
            copyto!(theta_m, theta); theta_m[r] -= h
            g[r] = _direct_coordinate_grad(cc, theta_p, theta_m, ctx, obj, lambda, arg0_base, h,
                Gp_buf[], Gm_buf[], linkp_buf[], linkm_buf[], profit_buf[],
                uplus_buf[], uminus_buf[], psi_buf[])
        end
        return nothing
    end
    return melitz_gradient_delta_direct_serial!
end

"""
    make_melitz_gradient_delta_direct_parallel(h) -> Function

Backend `:B_direct_argument_parallel`: the `Threads.@threads :static` coordinate sweep on
top of the direct serial backend above. `arg0_base` is shared, READ-ONLY, built once
BEFORE the parallel region (exactly the pattern `:B_localized_parallel` already
established for its own shared `Gbase`); each thread gets its own `(W, maxcols)`
probe-buffer pair and `O(W)` scratch, keyed by `Threads.maxthreadid()` (not
`Threads.nthreads()`, matching the other parallel backends' own documented rationale).
`g[r]` writes are disjoint per coordinate -- no synchronization needed. Same
guard/BLAS-thread discipline as `:B_localized_parallel`/`:B_argument_localized_parallel`.
"""
function make_melitz_gradient_delta_direct_parallel(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    arg0_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    Gp_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    Gm_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    linkp_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    linkm_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    profit_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    uplus_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    uminus_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    psi_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    nthreads_alloc = Ref(0)

    function melitz_gradient_delta_direct_parallel!(g::AbstractVector{Float64}, theta::AbstractVector{Float64},
                                                       ctx, obj, x::AbstractVector{Float64})
        n = length(theta)
        W = size(obj.U, 1)
        nt = Threads.maxthreadid()

        if compact_cache[] === nothing || ctx_cache[] !== ctx
            compact_cache[] = melitz_compact_columns_map(ctx)
            ctx_cache[] = ctx
        end
        compact = compact_cache[]
        maxcols = maximum(length(c.direct_cols) for c in compact)
        if arg0_buf[] === nothing || length(arg0_buf[]) != W ||
           nthreads_alloc[] != nt || size(Gp_bufs[][1]) != (W, maxcols)
            arg0_buf[] = zeros(Float64, W)
            Gp_bufs[] = [zeros(Float64, W, maxcols) for _ in 1:nt]
            Gm_bufs[] = [zeros(Float64, W, maxcols) for _ in 1:nt]
            linkp_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            linkm_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            profit_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            uplus_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            uminus_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            psi_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            nthreads_alloc[] = nt
        end
        arg0_base = arg0_buf[]
        _base_arg0!(arg0_base, obj, x)
        lambda = @view x[2:end]

        prev_blas_threads = BLAS.get_num_threads()
        guards_on = isdefined(Main, :CounterfactualSensitivity)
        BLAS.set_num_threads(1)
        guards_on && Main.CounterfactualSensitivity.guard_enter_coord_pool!()
        try
            Threads.@threads :static for r in 1:n
                tid = Threads.threadid()
                cc = compact[r]
                theta_p = copy(theta); theta_p[r] += h
                theta_m = copy(theta); theta_m[r] -= h
                g[r] = _direct_coordinate_grad(cc, theta_p, theta_m, ctx, obj, lambda, arg0_base, h,
                    Gp_bufs[][tid], Gm_bufs[][tid], linkp_bufs[][tid], linkm_bufs[][tid],
                    profit_bufs[][tid], uplus_bufs[][tid], uminus_bufs[][tid], psi_bufs[][tid])
            end
        finally
            guards_on && Main.CounterfactualSensitivity.guard_exit_coord_pool!()
            BLAS.set_num_threads(prev_blas_threads)
        end
        return nothing
    end
    return melitz_gradient_delta_direct_parallel!
end
