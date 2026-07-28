# Touched-row (no-full-copy) fixed-dual outer-gradient backend -- governing prompt Phase 4
# (2026-07-27 continuation, outer-search session).
#
# CONTEXT: `sorted_crossing_gradient.jl`'s crossing-slice backend already avoids computing
# `G`'s columns outside the sorted crossing slice, but its OWN per-coordinate body still
# (a) `copyto!(u_plus, arg0_base)`/`copyto!(u_minus, arg0_base)` -- a FULL W-length copy, and
# (b) calls `obj.Psi!(psi_buf, u_plus)` (and `u_minus`) -- a FULL W-length elementwise Psi
# evaluation -- even though only the touched rows (the crossing slice, or the whole column for
# a link-touching coordinate) actually differ from the cached base state. At real D=20
# (`W=80,000`, `n_theta=798`), this is the measured ~1.02GB/gradient-call memory-TRAFFIC cost
# documented in `docs/melitz_final_allocation_and_gradient_closure_2026-07-27.md` Phase 4 item
# 1 (`2*W*n_theta*8 bytes`) -- not an allocation hazard (destination buffers are already
# persistent/reused), but real bandwidth this backend removes.
#
# EXACT IDENTITY THIS BACKEND EXPLOITS: `Psi` is applied ELEMENTWISE
# (`melitz_cc_Psi!`, cc_bundle.jl) and the fixed-dual scalar objective is
# `sum(Psi.(u))/W` (up to the `+zeta`/`1e10`/sign conventions the caller already applies). For
# any row `w` NOT among this coordinate's touched rows, `u_plus[w]==u_minus[w]==u_base[w]`
# EXACTLY (the crossing-slice proof already in `sorted_crossing_gradient.jl`'s own header --
# this file adds no new dependency-map claim, it only changes HOW the same already-proved
# "untouched rows are unaffected" fact is exploited: skipping their Psi call entirely, not
# merely skipping their `G` column fill). So, writing `psi_base[w] = Psi(u_base[w])` and
# `base_scalar_sum = sum(psi_base)` (computed ONCE per gradient call, not per coordinate):
#
#     sum(Psi.(u_plus)) = base_scalar_sum + sum_{w touched} [Psi(u_plus[w]) - psi_base[w]]
#
# -- mathematically IDENTICAL to the full sum, evaluated only at touched rows.
#
# TOUCHED-ROW BOOKKEEPING (no full-W reset, ever): `touched_gen::Vector{Int}` (persistent,
# per-thread) + a scalar `gen` counter incremented once per coordinate probe. Row `w`'s
# `delta_plus[w]`/`delta_minus[w]` are FRESHLY WRITTEN (not accumulated) the first time `w` is
# touched in a given generation (`touched_gen[w] != gen`), and accumulated (`+=`) on any
# subsequent touch within the SAME generation (e.g. two direct columns sharing origin `o`
# whose crossing slices overlap). This means `delta_plus`/`delta_minus` need NEVER be reset
# between coordinates -- a stale value from 2+ generations ago is simply never read (guarded
# by the generation check), so the O(W) `fill!`/copy this design set out to eliminate does not
# reappear anywhere, including in the "reset" step.
#
# SCOPE (matches `sorted_crossing_gradient.jl`'s own established scope decision): direct
# trade-cell columns use the sparse touched-row machinery; the focal-link column (when
# `cc.touches_link`), which already touches all `W` rows by construction (`_fill_compact_link_
# from_state!` is dense), is handled by marking all `W` rows touched for that coordinate --
# correct (identical formula) but with no touched-row SAVING for link-touching coordinates,
# exactly as expected (there is nothing to save when every row is genuinely touched).
# Coefficient-only, cutoff-only, and mixed cutoff+coefficient directions are all covered
# uniformly (the crossing-slice contribution formula is unchanged from the existing backend;
# only the apply/evaluate step changes) -- these are properties of WHICH cells a coordinate's
# `MelitzCompactColumns` touches, already handled by the shared, unmodified dependency map.

"""
    _touch_row!(w, cp, cm, delta_plus, delta_minus, touched_gen, gen, touched_list)

Record that row `w` is touched in generation `gen` with contribution `cp`/`cm` (added to
`delta_plus[w]`/`delta_minus[w]`). First touch this generation: fresh write + push to
`touched_list`. Subsequent touch (same generation, e.g. an overlapping crossing slice from
another column): accumulate. Never reads/writes any row NOT touched this generation, and never
resets anything -- the generation stamp alone makes stale values from prior generations
unreachable.
"""
@inline function _touch_row!(w::Int, cp::Float64, cm::Float64,
                              delta_plus::Vector{Float64}, delta_minus::Vector{Float64},
                              touched_gen::Vector{Int}, gen::Int, touched_list::Vector{Int})
    @inbounds if touched_gen[w] != gen
        touched_gen[w] = gen
        delta_plus[w] = cp
        delta_minus[w] = cm
        push!(touched_list, w)
    else
        @inbounds delta_plus[w] += cp
        @inbounds delta_minus[w] += cm
    end
    return nothing
end

"Scalar form of `melitz_cc_Psi!`'s own elementwise map (cc_bundle.jl) -- identical formula, one value at a time, for the touched-row evaluate step below."
@inline function _touched_row_psi_scalar(u::Float64)
    return (u <= 1.0 ? exp(u) : (u^2 + 1.0) * 0.5 * exp(1)) - 1.0
end

"""
    _direct_coordinate_grad_touched_row(cc, theta_p, theta_m, ctx, obj, sorted_ctx, lambda,
        arg0_base, psi_base, base_scalar_sum, h, Gp, Gm, union_start, linkp, linkm, profit,
        delta_plus, delta_minus, touched_gen, gen, touched_list, state_p, state_m, ws) -> Float64

Touched-row analogue of `sorted_crossing_gradient.jl`'s `_direct_coordinate_grad_sorted`:
identical formula and identical numerical result (validated in `test/melitz/runtests.jl`), but
never copies `arg0_base` and never calls `Psi!` on an untouched row. `psi_base`/
`base_scalar_sum` are computed ONCE per gradient call by the caller (not per coordinate).
`touched_list` is `empty!`-ed by the caller's own `gen` bump convention (see below) -- this
function pushes onto it and the caller drains it after reading, an O(touched) operation.
"""
function _direct_coordinate_grad_touched_row(cc::MelitzCompactColumns, theta_p::AbstractVector{Float64},
                                              theta_m::AbstractVector{Float64}, ctx, obj, sorted_ctx::MelitzSortedTailContext,
                                              lambda::AbstractVector{Float64}, arg0_base::AbstractVector{Float64},
                                              psi_base::AbstractVector{Float64}, base_scalar_sum::Float64,
                                              h::Real, Gp::AbstractMatrix{Float64}, Gm::AbstractMatrix{Float64},
                                              union_start::AbstractVector{Int}, linkp::AbstractVector{Float64},
                                              linkm::AbstractVector{Float64}, profit::AbstractVector{Float64},
                                              delta_plus::Vector{Float64}, delta_minus::Vector{Float64},
                                              touched_gen::Vector{Int}, gen::Int, touched_list::Vector{Int},
                                              state_p::MelitzExpandedState, state_m::MelitzExpandedState,
                                              ws::MelitzThetaExpansionWorkspace)
    W = length(arg0_base)
    layout = ctx.moment_layout
    ncols = length(cc.direct_cols)
    empty!(touched_list)

    melitz_expand_theta!(state_p, theta_p, ctx, ws)
    melitz_expand_theta!(state_m, theta_m, ctx, ws)

    if ncols > 0
        _fill_compact_direct_columns_crossing_sorted!(Gp, Gm, union_start, ctx, sorted_ctx,
            cc.direct_cells, ncols, state_p, state_m)
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
        link_col_offset = 2 + layout.focal_link_index
        @inbounds for w in 1:W
            gbase = obj.H[w, link_col_offset]
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
    make_melitz_gradient_delta_direct_touched_row_serial(h) -> Function

Backend `:B_direct_argument_touched_row_serial`: same calling convention, same crossing-slice
`G`-column-fill cost as `:B_direct_argument_sorted_serial`, but replaces the per-coordinate
full-`W` `copyto!`+`Psi!` apply/evaluate step with the touched-row-only accumulate/evaluate
described in this file's header. Requires `ctx.sorted_tail_ctx !== nothing`, identical
requirement to the sorted backend it is checked against.
"""
function make_melitz_gradient_delta_direct_touched_row_serial(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    arg0_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    psi_base_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    Gp_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gm_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    union_start_buf = Ref{Union{Nothing,Vector{Int}}}(nothing)
    linkp_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    linkm_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    profit_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    delta_plus_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    delta_minus_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    touched_gen_buf = Ref{Union{Nothing,Vector{Int}}}(nothing)
    touched_list_buf = Ref{Union{Nothing,Vector{Int}}}(nothing)
    # Monotonically increasing ACROSS EVERY CALL to the returned closure, never reset to 1 --
    # using the per-call coordinate index `r` (1:n) directly as the generation stamp would
    # collide with a DIFFERENT gradient call's own generation `r` (e.g. coordinate 1 of call #2
    # reusing the stamp value coordinate 1 of call #1 already left in `touched_gen`), silently
    # reading stale `delta_plus`/`delta_minus` values from a previous call as if freshly
    # written. Caught before this backend was ever run, not found via a failing test.
    gen_counter = Ref(0)
    thetap_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    thetam_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    ws_buf = Ref{Union{Nothing,MelitzThetaExpansionWorkspace}}(nothing)
    statep_buf = Ref{Union{Nothing,MelitzExpandedState}}(nothing)
    statem_buf = Ref{Union{Nothing,MelitzExpandedState}}(nothing)

    function melitz_gradient_delta_direct_touched_row_serial!(g::AbstractVector{Float64}, theta::AbstractVector{Float64},
                                                                ctx, obj, x::AbstractVector{Float64})
        sorted_ctx = get(ctx, :sorted_tail_ctx, nothing)
        sorted_ctx === nothing && throw(ArgumentError(
            "melitz_gradient_delta_direct_touched_row_serial!: ctx.sorted_tail_ctx is nothing -- " *
            "build the bundle with moment_backend=:sorted_tail_serial or :sorted_tail_parallel " *
            "before selecting gradient_backend=:B_direct_argument_touched_row_serial"))
        n = length(theta)
        W = size(obj.U, 1)

        if compact_cache[] === nothing || ctx_cache[] !== ctx
            compact_cache[] = melitz_compact_columns_map(ctx)
            ctx_cache[] = ctx
            ws_buf[] = MelitzThetaExpansionWorkspace(ctx.D)
            statep_buf[] = MelitzExpandedState(ctx.D)
            statem_buf[] = MelitzExpandedState(ctx.D)
        end
        compact = compact_cache[]
        maxcols = maximum(length(c.direct_cols) for c in compact)
        if arg0_buf[] === nothing || length(arg0_buf[]) != W || size(Gp_buf[]) != (W, maxcols)
            arg0_buf[] = zeros(Float64, W)
            psi_base_buf[] = zeros(Float64, W)
            Gp_buf[] = zeros(Float64, W, maxcols)
            Gm_buf[] = zeros(Float64, W, maxcols)
            union_start_buf[] = zeros(Int, maxcols)
            linkp_buf[] = zeros(Float64, W)
            linkm_buf[] = zeros(Float64, W)
            profit_buf[] = zeros(Float64, W)
            delta_plus_buf[] = zeros(Float64, W)
            delta_minus_buf[] = zeros(Float64, W)
            touched_gen_buf[] = zeros(Int, W)
            touched_list_buf[] = sizehint!(Int[], 4 * ctx.D)
        end
        if thetap_buf[] === nothing || length(thetap_buf[]) != n
            thetap_buf[] = zeros(Float64, n)
            thetam_buf[] = zeros(Float64, n)
        end
        arg0_base = arg0_buf[]
        _base_arg0!(arg0_base, obj, x)
        psi_base = psi_base_buf[]
        @inbounds for w in 1:W
            psi_base[w] = _touched_row_psi_scalar(arg0_base[w])
        end
        base_scalar_sum = sum(psi_base)
        lambda = @view x[2:end]

        theta_p = thetap_buf[]
        theta_m = thetam_buf[]
        touched_gen = touched_gen_buf[]
        touched_list = touched_list_buf[]
        for r in 1:n
            cc = compact[r]
            copyto!(theta_p, theta); theta_p[r] += h
            copyto!(theta_m, theta); theta_m[r] -= h
            gen_counter[] += 1
            g[r] = _direct_coordinate_grad_touched_row(cc, theta_p, theta_m, ctx, obj, sorted_ctx, lambda,
                arg0_base, psi_base, base_scalar_sum, h,
                Gp_buf[], Gm_buf[], union_start_buf[], linkp_buf[], linkm_buf[], profit_buf[],
                delta_plus_buf[], delta_minus_buf[], touched_gen, gen_counter[], touched_list,
                statep_buf[], statem_buf[], ws_buf[])
        end
        return nothing
    end
    return melitz_gradient_delta_direct_touched_row_serial!
end

"""
    make_melitz_gradient_delta_direct_touched_row_parallel(h) -> Function

Backend `:B_direct_argument_touched_row_parallel`: `Threads.@threads :static` coordinate sweep
on top of the touched-row serial backend above, mirroring `sorted_crossing_gradient.jl`'s own
`make_melitz_gradient_delta_direct_sorted_parallel` pattern (thread-local buffers, one
generation counter PER THREAD since each thread owns its own `touched_gen` array -- no
cross-thread stamp collision is possible even without a shared counter). `arg0_base`/
`psi_base`/`base_scalar_sum` are shared, READ-ONLY across threads once built before the
parallel region (identical convention to the sorted-parallel backend's own shared `arg0_base`).

User-motivated addition (2026-07-XX): the serial touched-row backend measured only a 1.02x
wall-clock gain over the sorted backend despite a measured ~5.7x memory-traffic reduction --
a real dissociation suggesting memory BANDWIDTH is not the bottleneck for a single thread. This
parallel variant tests the different, actually-production-relevant hypothesis: with 20
threads simultaneously contending for shared DRAM bandwidth, a per-thread traffic reduction
that did nothing serially could plausibly become the binding constraint. See the benchmark
script for the direct measurement -- do not assume speedup transfers from the single-thread
result without checking.
"""
function make_melitz_gradient_delta_direct_touched_row_parallel(h::Real)
    compact_cache = Ref{Union{Nothing,Vector{MelitzCompactColumns}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    arg0_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    psi_base_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    Gp_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    Gm_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    union_start_bufs = Ref{Union{Nothing,Vector{Vector{Int}}}}(nothing)
    linkp_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    linkm_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    profit_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    delta_plus_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    delta_minus_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    touched_gen_bufs = Ref{Union{Nothing,Vector{Vector{Int}}}}(nothing)
    touched_list_bufs = Ref{Union{Nothing,Vector{Vector{Int}}}}(nothing)
    thetap_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    thetam_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    nthreads_alloc = Ref(0)
    ws_bufs = Ref{Union{Nothing,Vector{MelitzThetaExpansionWorkspace}}}(nothing)
    statep_bufs = Ref{Union{Nothing,Vector{MelitzExpandedState}}}(nothing)
    statem_bufs = Ref{Union{Nothing,Vector{MelitzExpandedState}}}(nothing)
    gen_counters = Ref{Union{Nothing,Vector{Int}}}(nothing)   # one per thread, never reset, mirrors the serial backend's own never-reset convention

    function melitz_gradient_delta_direct_touched_row_parallel!(g::AbstractVector{Float64}, theta::AbstractVector{Float64},
                                                                   ctx, obj, x::AbstractVector{Float64})
        sorted_ctx = get(ctx, :sorted_tail_ctx, nothing)
        sorted_ctx === nothing && throw(ArgumentError(
            "melitz_gradient_delta_direct_touched_row_parallel!: ctx.sorted_tail_ctx is nothing -- " *
            "build the bundle with moment_backend=:sorted_tail_serial or :sorted_tail_parallel " *
            "before selecting gradient_backend=:B_direct_argument_touched_row_parallel"))
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
            psi_base_buf[] = zeros(Float64, W)
            Gp_bufs[] = [zeros(Float64, W, maxcols) for _ in 1:nt]
            Gm_bufs[] = [zeros(Float64, W, maxcols) for _ in 1:nt]
            union_start_bufs[] = [zeros(Int, maxcols) for _ in 1:nt]
            linkp_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            linkm_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            profit_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            delta_plus_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            delta_minus_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            touched_gen_bufs[] = [zeros(Int, W) for _ in 1:nt]
            touched_list_bufs[] = [sizehint!(Int[], 4 * ctx.D) for _ in 1:nt]
            gen_counters[] = zeros(Int, nt)
            nthreads_alloc[] = nt
        end
        if thetap_bufs[] === nothing || length(thetap_bufs[]) != nt || length(thetap_bufs[][1]) != n
            thetap_bufs[] = [zeros(Float64, n) for _ in 1:nt]
            thetam_bufs[] = [zeros(Float64, n) for _ in 1:nt]
        end
        if ws_bufs[] === nothing || length(ws_bufs[]) != nt || ws_bufs[][1].D != ctx.D
            ws_bufs[] = [MelitzThetaExpansionWorkspace(ctx.D) for _ in 1:nt]
            statep_bufs[] = [MelitzExpandedState(ctx.D) for _ in 1:nt]
            statem_bufs[] = [MelitzExpandedState(ctx.D) for _ in 1:nt]
        end
        arg0_base = arg0_buf[]
        _base_arg0!(arg0_base, obj, x)
        psi_base = psi_base_buf[]
        @inbounds for w in 1:W
            psi_base[w] = _touched_row_psi_scalar(arg0_base[w])
        end
        base_scalar_sum = sum(psi_base)
        lambda = @view x[2:end]

        prev_blas_threads = BLAS.get_num_threads()
        guards_on = isdefined(Main, :CounterfactualSensitivity)
        BLAS.set_num_threads(1)
        guards_on && Main.CounterfactualSensitivity.guard_enter_coord_pool!()
        try
            Threads.@threads :static for r in 1:n
                tid = Threads.threadid()
                cc = compact[r]
                theta_p = thetap_bufs[][tid]; copyto!(theta_p, theta); theta_p[r] += h
                theta_m = thetam_bufs[][tid]; copyto!(theta_m, theta); theta_m[r] -= h
                gen_counters[][tid] += 1
                g[r] = _direct_coordinate_grad_touched_row(cc, theta_p, theta_m, ctx, obj, sorted_ctx, lambda,
                    arg0_base, psi_base, base_scalar_sum, h,
                    Gp_bufs[][tid], Gm_bufs[][tid], union_start_bufs[][tid], linkp_bufs[][tid], linkm_bufs[][tid],
                    profit_bufs[][tid], delta_plus_bufs[][tid], delta_minus_bufs[][tid],
                    touched_gen_bufs[][tid], gen_counters[][tid], touched_list_bufs[][tid],
                    statep_bufs[][tid], statem_bufs[][tid], ws_bufs[][tid])
            end
        finally
            guards_on && Main.CounterfactualSensitivity.guard_exit_coord_pool!()
            BLAS.set_num_threads(prev_blas_threads)
        end
        return nothing
    end
    return melitz_gradient_delta_direct_touched_row_parallel!
end
