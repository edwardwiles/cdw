# Sorted crossing-slice outer-gradient backend (2026-07-25 session continuation, Phase 6/7
# of docs/melitz_sorted_tail_optimization_2026-07-25.md). See that report's Section D.1 for
# the full derivation and validation results.
#
# CONTEXT: the production registered outer gradient (`direct_gradient.jl`'s
# `:B_direct_argument_serial`/`_parallel`) already restricts its per-coordinate probe to a
# SMALL set of `O(D)` "touched" trade cells (`argument_localized_gradient.jl`'s
# `melitz_compact_columns_map`, reusing the already-validated Gate 1/2 dependency audit --
# this session's own work does NOT re-derive or re-audit which cells are touched by which
# coordinate, only how each touched column's W-length values get computed). But
# `_fill_compact_direct_columns!` still does a FULL dense `W`-row scan per touched column,
# for BOTH `theta+h` and `theta-h` -- exactly the same O(W) pattern Phase 3 already
# eliminated for the full moment matrix, just re-appearing at the gradient's own per-column
# probe.
#
# THE CROSSING-SLICE ARGUMENT (proved exactly, not merely conjectured):
#
# `affine_cutoff.jl`'s own already-validated result: `q_od(theta) = q0_od + Q[od,:].theta`
# is EXACTLY AFFINE in `theta_free`. A central-difference probe perturbs a single coordinate
# `r` by `+h`/`-h` symmetrically, so
#     q_od(theta+h*e_r) = q_od(theta) + h*Q[od,r]
#     q_od(theta-h*e_r) = q_od(theta) - h*Q[od,r]
# i.e. `q_od(theta)` (the BASE point's own log-cutoff) is EXACTLY the midpoint, in LOG space,
# of the plus/minus probe's own log-cutoffs. Since `log` is monotone, `cutoff_od(theta)`
# (the base cutoff, in LEVELS) therefore lies WEAKLY BETWEEN `cutoff_od(theta+h*e_r)` and
# `cutoff_od(theta-h*e_r)` in levels too: `min(cutoff_p,cutoff_m) <= cutoff_base <=
# max(cutoff_p,cutoff_m)`. Because `melitz_active_tail_start` is monotone nondecreasing in
# its cutoff argument, the same weak ordering holds for the active-tail START POSITIONS:
# `min(k_p,k_m) <= k_base <= max(k_p,k_m)`.
#
# Consequence: for any sorted position `pos < min(k_p,k_m)`, the draw at that position is
# INACTIVE under theta+h*e_r, theta-h*e_r, AND theta (the base point) -- all three,
# simultaneously, proved (not assumed) by the sandwich above. At such a position, `Gp = Gm =
# gbase = -lambda_od` EXACTLY (the same inactive base value in all three evaluations), so
# `Gp-gbase` and `Gm-gbase` are EXACTLY zero and contribute nothing to `u_plus`/`u_minus`
# (`direct_gradient.jl`) or to `G_jac[:,gcol,k]` (`argument_localized_gradient.jl`). Only
# the CROSSING SLICE `min(k_p,k_m):W` -- not the full `1:W` -- ever needs computing or
# applying. This is an EXACT algebraic identity, not a numerical approximation: the affine
# cutoff structure this codebase already builds and validates elsewhere (`affine_cutoff.jl`)
# is what makes it exact rather than merely "usually true nearby."
#
# Scope: implemented for the DIRECT trade cells only (mirrors Phase 3's own C.1 scope
# decision) -- the focal link column (`cc.touches_link`) is still filled DENSELY, reusing
# `argument_localized_gradient.jl`'s existing `_fill_compact_link!` unchanged. New backend
# `:B_direct_argument_sorted_serial`, gated behind `ctx.sorted_tail_ctx !== nothing`
# (requires the bundle to have been built with `moment_backend` in
# `(:sorted_tail_serial,:sorted_tail_parallel)`) -- never silently falls back to a stale or
# absent sorted context. Does NOT modify `:B_direct_argument_serial`/`_parallel` or
# `:B_argument_localized_serial`/`_parallel` at all; those remain the validated reference
# this new backend is checked against.

using LinearAlgebra: BLAS

"""
    _fill_compact_direct_columns_crossing_sorted!(Gp, Gm, union_start, theta_p, theta_m,
                                                    ctx, sorted_ctx, direct_cells, ncols)

Fills `Gp`/`Gm` (ORIGINAL-row indexed, `(W,>=ncols)` buffers) ONLY at the sorted crossing
slice `union_start[idx]:W` for each cell `direct_cells[idx]` (scattered back to original row
indices via `sorted_ctx.permutation`) -- rows outside that slice are NOT written (see this
file's own header for the exact proof that they are provably unneeded: inactive at `theta_p`,
`theta_m`, AND the base point simultaneously). `union_start[idx] = min(k_p,k_m)`, the sorted
position where either displaced cutoff's active tail begins, is written into the
caller-owned `union_start` vector for the caller's own crossing-slice apply step.
"""
function _fill_compact_direct_columns_crossing_sorted!(Gp::AbstractMatrix{Float64}, Gm::AbstractMatrix{Float64},
                                                         union_start::AbstractVector{Int},
                                                         theta_p::AbstractVector{Float64}, theta_m::AbstractVector{Float64},
                                                         ctx, sorted_ctx::MelitzSortedTailContext,
                                                         direct_cells::Vector{Tuple{Int,Int}}, ncols::Int)
    sigma = ctx.sigma
    W = sorted_ctx.W
    Ap, fp, _, _ = melitz_expand_theta(theta_p, ctx)
    Am, fm, _, _ = melitz_expand_theta(theta_m, ctx)
    @inbounds for idx in 1:ncols
        (o, d) = direct_cells[idx]
        lambda_od = ctx.X_data[o, d] / ctx.expenditure[d]
        Cp_od = melitz_C(ctx.w[o], ctx.tau[o, d], Ap[o, d], sigma, ctx.expenditure[d])
        Cm_od = melitz_C(ctx.w[o], ctx.tau[o, d], Am[o, d], sigma, ctx.expenditure[d])
        cutoff_p = melitz_cutoff(ctx.w[o], fp[o, d], sigma, Cp_od)
        cutoff_m = melitz_cutoff(ctx.w[o], fm[o, d], sigma, Cm_od)
        coef_p = Cp_od / ctx.expenditure[d]
        coef_m = Cm_od / ctx.expenditure[d]
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        perm_o = @view sorted_ctx.permutation[:, o]
        zpow_o = @view sorted_ctx.sorted_z_power[:, o]
        kp = melitz_active_tail_start(sorted_z_o, cutoff_p)
        km = melitz_active_tail_start(sorted_z_o, cutoff_m)
        kunion = min(kp, km)
        union_start[idx] = kunion
        for pos in kunion:W
            s = perm_o[pos]
            Gp[s, idx] = pos >= kp ? coef_p * zpow_o[pos] - lambda_od : -lambda_od
            Gm[s, idx] = pos >= km ? coef_m * zpow_o[pos] - lambda_od : -lambda_od
        end
    end
    return union_start
end

"""
    _direct_coordinate_grad_sorted(cc, theta_p, theta_m, ctx, obj, sorted_ctx, lambda,
                                    arg0_base, h, Gp, Gm, union_start, linkp, linkm, profit,
                                    u_plus, u_minus, psi_buf) -> Float64

Crossing-slice analogue of `direct_gradient.jl`'s `_direct_coordinate_grad`: identical
formula and identical `u_plus`/`u_minus` CONTENTS (this file's header proves every skipped
row contributes exactly zero), but the direct-cell fill AND the `u_plus`/`u_minus` apply
loop both run only over each column's own crossing slice (scattered via
`sorted_ctx.permutation`) instead of a dense `1:W` scan. The focal link column, when
touched, is still filled DENSELY via the unchanged `_fill_compact_link!` -- Section C.1's
own scope decision, reapplied here.
"""
function _direct_coordinate_grad_sorted(cc::MelitzCompactColumns, theta_p::AbstractVector{Float64},
                                         theta_m::AbstractVector{Float64}, ctx, obj, sorted_ctx::MelitzSortedTailContext,
                                         lambda::AbstractVector{Float64}, arg0_base::AbstractVector{Float64},
                                         h::Real, Gp::AbstractMatrix{Float64}, Gm::AbstractMatrix{Float64},
                                         union_start::AbstractVector{Int}, linkp::AbstractVector{Float64},
                                         linkm::AbstractVector{Float64}, profit::AbstractVector{Float64},
                                         u_plus::AbstractVector{Float64}, u_minus::AbstractVector{Float64},
                                         psi_buf::AbstractVector{Float64})
    W = length(arg0_base)
    layout = ctx.moment_layout
    ncols = length(cc.direct_cols)

    copyto!(u_plus, arg0_base)
    copyto!(u_minus, arg0_base)

    if ncols > 0
        _fill_compact_direct_columns_crossing_sorted!(Gp, Gm, union_start, theta_p, theta_m,
            ctx, sorted_ctx, cc.direct_cells, ncols)
        @inbounds for idx in 1:ncols
            gcol = cc.direct_cols[idx]
            lam_k = lambda[gcol]
            Gbase_col_offset = 2 + gcol
            (o, _d) = cc.direct_cells[idx]
            perm_o = @view sorted_ctx.permutation[:, o]
            kunion = union_start[idx]
            for pos in kunion:W
                w = perm_o[pos]
                gbase = obj.H[w, Gbase_col_offset]
                u_plus[w] -= lam_k * (Gp[w, idx] - gbase)
                u_minus[w] -= lam_k * (Gm[w, idx] - gbase)
            end
        end
    end

    if cc.touches_link
        _fill_compact_link!(linkp, profit, theta_p, ctx, obj)
        _fill_compact_link!(linkm, profit, theta_m, ctx, obj)
        lam_link = lambda[layout.focal_link_index]
        link_col_offset = 2 + layout.focal_link_index
        @inbounds for w in 1:W
            gbase = obj.H[w, link_col_offset]
            u_plus[w] -= lam_link * (linkp[w] - gbase)
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
    make_melitz_gradient_delta_direct_sorted_serial(h) -> Function

Backend `:B_direct_argument_sorted_serial`: same calling convention
(`(g,theta,ctx,obj,x)->nothing`) and same central-difference construction as
`:B_direct_argument_serial`, but every touched direct trade column is computed and applied
only over its own sorted crossing slice (this file's header). Requires
`ctx.sorted_tail_ctx !== nothing` -- throws `ArgumentError` immediately (before any KNITRO
callback work) if the bundle was not built with `moment_backend` in
`(:sorted_tail_serial,:sorted_tail_parallel)`, rather than silently falling back to a dense
scan or a stale/absent context.
"""
function make_melitz_gradient_delta_direct_sorted_serial(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    arg0_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    Gp_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gm_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    union_start_buf = Ref{Union{Nothing,Vector{Int}}}(nothing)
    linkp_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    linkm_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    profit_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    uplus_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    uminus_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    psi_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    thetap_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    thetam_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)

    function melitz_gradient_delta_direct_sorted_serial!(g::AbstractVector{Float64}, theta::AbstractVector{Float64},
                                                           ctx, obj, x::AbstractVector{Float64})
        sorted_ctx = get(ctx, :sorted_tail_ctx, nothing)
        sorted_ctx === nothing && throw(ArgumentError(
            "melitz_gradient_delta_direct_sorted_serial!: ctx.sorted_tail_ctx is nothing -- " *
            "build the bundle with moment_backend=:sorted_tail_serial or :sorted_tail_parallel " *
            "before selecting gradient_backend=:B_direct_argument_sorted_serial"))
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
            union_start_buf[] = zeros(Int, maxcols)
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
            g[r] = _direct_coordinate_grad_sorted(cc, theta_p, theta_m, ctx, obj, sorted_ctx, lambda, arg0_base, h,
                Gp_buf[], Gm_buf[], union_start_buf[], linkp_buf[], linkm_buf[], profit_buf[],
                uplus_buf[], uminus_buf[], psi_buf[])
        end
        return nothing
    end
    return melitz_gradient_delta_direct_sorted_serial!
end

"""
    make_melitz_gradient_delta_direct_sorted_parallel(h) -> Function

Backend `:B_direct_argument_sorted_parallel`: the `Threads.@threads :static` coordinate
sweep on top of the sorted crossing-slice serial backend above -- same thread-safety
discipline as `direct_gradient.jl`'s own `:B_direct_argument_parallel` (shared, READ-ONLY
`arg0_base` built once BEFORE the parallel region; `g[r]` writes disjoint per coordinate,
no synchronization needed; thread-local `(W,maxcols)` probe buffers AND a thread-local
`union_start` vector, keyed by `Threads.maxthreadid()`, not `Threads.nthreads()`; BLAS
forced to 1 thread for the sweep and restored via `try/finally`;
`cc_algo/parallelism_guards.jl`'s `guard_enter_coord_pool!`/`guard_exit_coord_pool!` reused;
no inner KNITRO solve reachable from a coordinate thread). Requires
`ctx.sorted_tail_ctx !== nothing`, same as the serial variant.
"""
function make_melitz_gradient_delta_direct_sorted_parallel(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    arg0_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    Gp_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    Gm_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    union_start_bufs = Ref{Union{Nothing,Vector{Vector{Int}}}}(nothing)
    linkp_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    linkm_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    profit_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    uplus_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    uminus_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    psi_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    thetap_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    thetam_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    nthreads_alloc = Ref(0)

    function melitz_gradient_delta_direct_sorted_parallel!(g::AbstractVector{Float64}, theta::AbstractVector{Float64},
                                                             ctx, obj, x::AbstractVector{Float64})
        sorted_ctx = get(ctx, :sorted_tail_ctx, nothing)
        sorted_ctx === nothing && throw(ArgumentError(
            "melitz_gradient_delta_direct_sorted_parallel!: ctx.sorted_tail_ctx is nothing -- " *
            "build the bundle with moment_backend=:sorted_tail_serial or :sorted_tail_parallel " *
            "before selecting gradient_backend=:B_direct_argument_sorted_parallel"))
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
            union_start_bufs[] = [zeros(Int, maxcols) for _ in 1:nt]
            linkp_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            linkm_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            profit_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            uplus_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            uminus_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            psi_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            nthreads_alloc[] = nt
        end
        # NOTE: checks `length(thetap_bufs[]) != nt` directly -- see direct_gradient.jl's
        # identical fix for why a shared `nthreads_alloc[] != nt` check here would be silently
        # defeated by the arg0_buf/Gp_bufs block above (which updates `nthreads_alloc[]` first).
        if thetap_bufs[] === nothing || length(thetap_bufs[]) != nt || length(thetap_bufs[][1]) != n
            thetap_bufs[] = [zeros(Float64, n) for _ in 1:nt]
            thetam_bufs[] = [zeros(Float64, n) for _ in 1:nt]
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
                # 2026-07-27 continuation (Phase 3.2): per-thread persistent theta_p/theta_m
                # buffers replace a `copy(theta)` allocation on EVERY coordinate -- see
                # direct_gradient.jl's identical fix for the full rationale.
                theta_p = thetap_bufs[][tid]; copyto!(theta_p, theta); theta_p[r] += h
                theta_m = thetam_bufs[][tid]; copyto!(theta_m, theta); theta_m[r] -= h
                g[r] = _direct_coordinate_grad_sorted(cc, theta_p, theta_m, ctx, obj, sorted_ctx, lambda, arg0_base, h,
                    Gp_bufs[][tid], Gm_bufs[][tid], union_start_bufs[][tid], linkp_bufs[][tid], linkm_bufs[][tid],
                    profit_bufs[][tid], uplus_bufs[][tid], uminus_bufs[][tid], psi_bufs[][tid])
            end
        finally
            guards_on && Main.CounterfactualSensitivity.guard_exit_coord_pool!()
            BLAS.set_num_threads(prev_blas_threads)
        end
        return nothing
    end
    return melitz_gradient_delta_direct_sorted_parallel!
end
