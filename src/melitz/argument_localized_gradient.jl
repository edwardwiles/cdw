# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 3: the fixed-dual ARGUMENT-localized gradient backend.
#
# -- Why this exists on top of the already-validated `:B_localized`/`:B_localized_parallel`
# (`localized_gradient.jl`) --
#
# Both existing localized backends already restrict the ECONOMIC computation (the
# `melitz_firm` calls) to each coordinate's own small `dep.cells` set via
# `fixed_active_set_moments_restricted!`. But BOTH still pay an O(W x K) cost PER
# COORDINATE that has nothing to do with economics: `copyto!(Gp, Gbase)` /
# `copyto!(Gm, Gbase)`, two full `(W, K)` matrix copies, done purely so the restricted fill's
# own contract ("G must already hold correct values at every untouched column") is
# satisfied, and so the final `(Gp .- Gm) ./ (2h)` subtraction gives EXACTLY zero at every
# untouched column (via `Gbase - Gbase`, not because either buffer independently "knows" the
# true derivative is zero there). Section 2's own memory audit
# (`docs/melitz_optimization_report_2026-07-23_continuation3.md`) shows this cost is
# O(W*K) per coordinate, O(W*K*n) per gradient call -- the dominant term once `D` (hence
# `K = D^2+1`) grows, projected to become the binding cost at D=10/D=20 production scale.
#
# The key realization this backend acts on: since the dependency map already PROVES (Gate 1,
# `melitz_localized_dependency_map`'s own validated superset claim) that every OTHER column
# is exactly invariant to coordinate `k`, `G_jac[:, othercol, k]` is EXACTLY `0.0` REGARDLESS
# of what `Gbase` happens to contain -- `Gbase` is never actually load-bearing for the
# OUTPUT, only for the cancellation mechanism. So this backend never builds `Gbase` (or any
# other full `(W, K)` matrix) AT ALL: `G_jac` is zeroed ONCE (a single vectorized memset, not
# an economic computation), and each coordinate writes ONLY its own touched columns, computed
# directly into small `(W, ncols_k)` buffers where `ncols_k = O(D)` (bounded by
# `|dep.cells| + D` when the coordinate touches the focal-link column, `O(1)` otherwise) --
# never `O(K) = O(D^2)`. Thread-local memory for the parallel variant is therefore `O(W*D)`
# in the worst case (link-touching coordinates), not `O(W*K)` -- the reduction the main
# prompt's own Section 3 asks for grows with `D` exactly because `D` grows linearly while
# `K = D^2+1` grows quadratically (D=4: ~7/17 columns touched at worst vs D=20: ~23/401).
#
# Design mirrors the SAME correctness discipline `localized_gradient.jl` already
# established: reuse `melitz_expand_theta`/`melitz_firm` (the identical pure functions the
# dense reference and both existing localized backends call), same floating-point operation
# ORDER for the profit_j accumulation (a fixed `d in 1:D` sweep, independent of storage
# order -- see `_fill_compact_link_and_profit!`'s own docstring for why this matters), so
# every touched column is BIT-IDENTICAL to `:B_localized`'s own output, not merely
# `isapprox`. Validated in the test suite ("Section 3: argument-localized gradient").

using LinearAlgebra: BLAS

"""
    MelitzCompactColumns

One free coordinate's LOCAL column layout for the argument-localized backend:
`direct_cells`/`direct_cols` (theta-independent, built once from `melitz_pivot_map`) list
every physical `(o,d)` trade cell this coordinate's own probe must recompute -- exactly
`dep.cells` UNIONED with the `D` origin-`j` destination cells whenever `touches_link` (the
SAME cells `fixed_active_set_moments_restricted!`'s `compute_link=true` branch already
always recomputes fresh, per its own docstring) -- deduplicated, in a FIXED but otherwise
arbitrary order (order does not affect correctness here: unlike `profit_j`, no column value
is an ACCUMULATOR, so recomputing an overlapping cell twice, if it ever happened, would give
the bit-identical deterministic value either way; in practice the union is already
deduplicated so this never happens).
"""
struct MelitzCompactColumns
    direct_cells::Vector{Tuple{Int,Int}}
    direct_cols::Vector{Int}     # global moment-column index for each entry in direct_cells
    touches_link::Bool
end

"""
    melitz_compact_columns_map(ctx) -> Vector{MelitzCompactColumns}

Builds one `MelitzCompactColumns` per free coordinate directly from
`melitz_localized_dependency_map(ctx)` (Gate 1, already validated by the `:B_localized`
test battery) -- this function adds NO new dependency claims of its own, it only
materializes the GLOBAL COLUMN layout implied by the existing, already-validated
`MelitzCoordDependency.cells`/`.touches_link` claim. Theta-independent, built once per
`ctx` (cached in the backend closures below, keyed by `ctx` object identity, exactly like
`depmap_cache` in `localized_gradient.jl`).
"""
function melitz_compact_columns_map(ctx)
    deps = melitz_localized_dependency_map(ctx)
    D, j = ctx.D, ctx.target_country
    layout = ctx.moment_layout
    out = Vector{MelitzCompactColumns}(undef, length(deps))
    for (k, dep) in enumerate(deps)
        cells = copy(dep.cells)
        if dep.touches_link
            for d in 1:D
                c = (j, d)
                c in cells || push!(cells, c)
            end
        end
        cols = [layout.trade_index[o, d] for (o, d) in cells]
        out[k] = MelitzCompactColumns(cells, cols, dep.touches_link)
    end
    return out
end

"""
    _fill_compact_direct_columns!(Gcols, theta_free, ctx, obj, direct_cells, ncols) -> Gcols

Fills `Gcols[:, 1:ncols]` (a `(W, >=ncols)` buffer, only the first `ncols` columns are
touched/meaningful -- callers reuse a single oversized buffer across coordinates of
different `ncols` rather than reallocating) with the trade-moment value
`realized_revenue/expenditure_d - lambda_od` for each of `direct_cells[1:ncols]`, in that
order. Identical per-cell formula/argument order to `_fill_fixed_active_set_moments!`'s own
cell loop (`gradient_lab.jl`) -- bit-identical output at any shared `(o,d,theta)` input.
"""
function _fill_compact_direct_columns!(Gcols::AbstractMatrix{Float64}, theta_free::AbstractVector{Float64},
                                        ctx, obj, direct_cells::Vector{Tuple{Int,Int}}, ncols::Int)
    sigma = ctx.sigma
    W = size(obj.U, 1)
    A, f, _gamma_prime_j, _f_jj = melitz_expand_theta(theta_free, ctx)
    price_power_d = 1.0
    @inbounds for idx in 1:ncols
        (o, d) = direct_cells[idx]
        lambda_od = ctx.X_data[o, d] / ctx.expenditure[d]
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma,
                                ctx.expenditure[d], price_power_d, z)
            Gcols[w, idx] = firm.realized_revenue / ctx.expenditure[d] - lambda_od
        end
    end
    return Gcols
end

"""
    _fill_compact_link!(link_col, profit_buf, theta_free, ctx, obj) -> link_col

Fills `link_col` (length `W`) with the focal-link moment value `g_free_entry_link` at each
draw, using `profit_buf` (length `W`, caller-owned scratch, overwritten) to accumulate
`profit_j` over `d in 1:D` origin-`j` destinations -- the EXACT SAME fixed order
`_fill_fixed_active_set_moments!`'s own `compute_link=true` branch uses (floating-point
addition is not associative; this order must match for bit-exactness, per that function's
own comment). Independent of `_fill_compact_direct_columns!` -- whether a given origin-`j`
destination cell ALSO appears in `direct_cells` is irrelevant here (this loop recomputes
`profit_j` fresh regardless, exactly mirroring the reference).
"""
function _fill_compact_link!(link_col::AbstractVector{Float64}, profit_buf::AbstractVector{Float64},
                              theta_free::AbstractVector{Float64}, ctx, obj)
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta_free, ctx)
    price_power_d = 1.0
    fill!(profit_buf, 0.0)
    @inbounds for d in 1:D
        lambda_od = ctx.X_data[j, d] / ctx.expenditure[d]
        for w in 1:W
            z = obj.U[w, j]
            firm = melitz_firm(ctx.w[j], ctx.tau[j, d], A[j, d], f[j, d], sigma,
                                ctx.expenditure[d], price_power_d, z)
            profit_buf[w] += firm.realized_operating_profit
        end
    end
    expenditure_prime = ctx.w_prime * ctx.L[j]
    price_power_autarky = gamma_prime_j
    @inbounds for w in 1:W
        z_j = obj.U[w, j]
        firm_auk = melitz_firm(ctx.w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, price_power_autarky, z_j)
        link_col[w] = profit_buf[w] / ctx.w[j] - firm_auk.realized_operating_profit / ctx.w_prime
    end
    return link_col
end

"""
    make_melitz_moments_jacobian_b_argument_localized_serial(h) -> Function

Backend `:B_argument_localized_serial`: same calling convention, dependency map, and
central-difference construction as `:B_localized`, but never builds `Gbase` or any full
`(W, K)` displaced-moment matrix. `G_jac` is zeroed once (a memset, not an economic
computation); each coordinate's own probe writes ONLY its touched columns
(`MelitzCompactColumns`), computed via `_fill_compact_direct_columns!`/`_fill_compact_link!`
into small reusable `(W, maxcols)` buffers (`maxcols` = the largest `ncols` across all
coordinates, precomputed once from the compact-column map). Requires BIT-IDENTICAL
agreement with `:B_localized` at every touched column (validated in the test suite).
"""
function make_melitz_moments_jacobian_b_argument_localized_serial(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    Gp_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gm_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    linkp_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    linkm_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    profit_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)

    function melitz_moments_jacobian_b_argument_localized_serial!(K_jac, G_jac, theta, U, obj)
        n = length(theta)
        K_jac .= 0.0
        K_jac[:, 1] .= 1.0
        if size(U, 1) < size(obj.U, 1)
            G_jac .= 0.0
            return nothing
        end
        ctx = obj.γ
        W = size(obj.U, 1)
        layout = ctx.moment_layout

        if compact_cache[] === nothing || ctx_cache[] !== ctx
            compact_cache[] = melitz_compact_columns_map(ctx)
            ctx_cache[] = ctx
        end
        compact = compact_cache[]
        maxcols = maximum(length(c.direct_cols) for c in compact)
        if Gp_buf[] === nothing || size(Gp_buf[]) != (W, maxcols)
            Gp_buf[] = zeros(Float64, W, maxcols)
            Gm_buf[] = zeros(Float64, W, maxcols)
            linkp_buf[] = zeros(Float64, W)
            linkm_buf[] = zeros(Float64, W)
            profit_buf[] = zeros(Float64, W)
        end
        Gp, Gm = Gp_buf[], Gm_buf[]
        linkp, linkm, profit = linkp_buf[], linkm_buf[], profit_buf[]

        fill!(G_jac, 0.0)

        ei = zeros(n)
        for k in 1:n
            cc = compact[k]
            ncols = length(cc.direct_cols)
            ei[k] = 1.0
            theta_p = theta .+ h .* ei
            theta_m = theta .- h .* ei

            _fill_compact_direct_columns!(Gp, theta_p, ctx, obj, cc.direct_cells, ncols)
            _fill_compact_direct_columns!(Gm, theta_m, ctx, obj, cc.direct_cells, ncols)
            @inbounds for idx in 1:ncols
                gcol = cc.direct_cols[idx]
                @views G_jac[:, gcol, k] .= (Gp[:, idx] .- Gm[:, idx]) ./ (2h)
            end

            if cc.touches_link
                _fill_compact_link!(linkp, profit, theta_p, ctx, obj)
                _fill_compact_link!(linkm, profit, theta_m, ctx, obj)
                @views G_jac[:, layout.focal_link_index, k] .= (linkp .- linkm) ./ (2h)
            end
            ei[k] = 0.0
        end
        return nothing
    end
    return melitz_moments_jacobian_b_argument_localized_serial!
end

"""
    make_melitz_moments_jacobian_b_argument_localized_parallel(h) -> Function

Backend `:B_argument_localized_parallel`: the `Threads.@threads :static` coordinate sweep
on top of the argument-localized serial backend above -- same thread-safety discipline as
`:B_localized_parallel` (`localized_gradient.jl`'s own header: base state built before the
parallel region if any existed, `G_jac[:, :, k]` disjoint per-`k` writes need no
synchronization, BLAS forced to 1 thread for the sweep and restored via `try/finally`,
`cc_algo/parallelism_guards.jl`'s `guard_enter_coord_pool!`/`guard_exit_coord_pool!` reused,
no inner KNITRO solve reachable from a coordinate thread, no global RNG mutation) -- except
there IS no shared base state to build here at all (this backend never constructs `Gbase`),
so the parallel region begins immediately after `G_jac` is zeroed. Thread-local scratch
(`Gp_bufs`/`Gm_bufs`/link/profit buffers, one per `Threads.maxthreadid()` slot, NOT
`Threads.nthreads()` -- see `localized_gradient.jl`'s own header for why that distinction is
load-bearing) is sized `(W, maxcols)` where `maxcols = O(D)`, not `(W, K) = O(D^2)` --
the memory reduction Section 3 of the governing prompt asks for.
"""
function make_melitz_moments_jacobian_b_argument_localized_parallel(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    Gp_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    Gm_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    linkp_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    linkm_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    profit_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    nthreads_alloc = Ref(0)

    function melitz_moments_jacobian_b_argument_localized_parallel!(K_jac, G_jac, theta, U, obj)
        n = length(theta)
        K_jac .= 0.0
        K_jac[:, 1] .= 1.0
        if size(U, 1) < size(obj.U, 1)
            G_jac .= 0.0
            return nothing
        end
        ctx = obj.γ
        W = size(obj.U, 1)
        layout = ctx.moment_layout
        nt = Threads.maxthreadid()

        if compact_cache[] === nothing || ctx_cache[] !== ctx
            compact_cache[] = melitz_compact_columns_map(ctx)
            ctx_cache[] = ctx
        end
        compact = compact_cache[]
        maxcols = maximum(length(c.direct_cols) for c in compact)
        if Gp_bufs[] === nothing || nthreads_alloc[] != nt || size(Gp_bufs[][1]) != (W, maxcols)
            Gp_bufs[] = [zeros(Float64, W, maxcols) for _ in 1:nt]
            Gm_bufs[] = [zeros(Float64, W, maxcols) for _ in 1:nt]
            linkp_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            linkm_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            profit_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            nthreads_alloc[] = nt
        end

        fill!(G_jac, 0.0)

        prev_blas_threads = BLAS.get_num_threads()
        guards_on = isdefined(Main, :CounterfactualSensitivity)
        BLAS.set_num_threads(1)
        guards_on && Main.CounterfactualSensitivity.guard_enter_coord_pool!()
        try
            Threads.@threads :static for k in 1:n
                tid = Threads.threadid()
                Gp, Gm = Gp_bufs[][tid], Gm_bufs[][tid]
                linkp, linkm, profit = linkp_bufs[][tid], linkm_bufs[][tid], profit_bufs[][tid]
                cc = compact[k]
                ncols = length(cc.direct_cols)
                theta_p = copy(theta)
                theta_p[k] += h
                theta_m = copy(theta)
                theta_m[k] -= h

                _fill_compact_direct_columns!(Gp, theta_p, ctx, obj, cc.direct_cells, ncols)
                _fill_compact_direct_columns!(Gm, theta_m, ctx, obj, cc.direct_cells, ncols)
                @inbounds for idx in 1:ncols
                    gcol = cc.direct_cols[idx]
                    @views G_jac[:, gcol, k] .= (Gp[:, idx] .- Gm[:, idx]) ./ (2h)
                end

                if cc.touches_link
                    _fill_compact_link!(linkp, profit, theta_p, ctx, obj)
                    _fill_compact_link!(linkm, profit, theta_m, ctx, obj)
                    @views G_jac[:, layout.focal_link_index, k] .= (linkp .- linkm) ./ (2h)
                end
            end
        finally
            guards_on && Main.CounterfactualSensitivity.guard_exit_coord_pool!()
            BLAS.set_num_threads(prev_blas_threads)
        end
        return nothing
    end
    return melitz_moments_jacobian_b_argument_localized_parallel!
end
