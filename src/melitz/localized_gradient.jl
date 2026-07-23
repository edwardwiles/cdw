using LinearAlgebra: BLAS

# Phase II.11 (screening-continuation session): localized fixed-dual gradient backend.
#
# GATE 1 (this section): the affected-moment-column dependency map, built and validated
# ALONE, with no gradient logic on top of it yet (main prompt Section 7's own required
# order: "build and unit-test the dependency map ALONE... verify its claimed affected-column
# set is a SUPERSET of the columns that actually differ under a finite perturbation, for
# every coordinate, at D=4" -- BEFORE any gradient code is written on top of it).
#
# -- Derivation (this session, generalizing/correcting the immediately-prior session's own
# sketch, which named the direct/pivot/focal-link dependencies but did not work through two
# subtleties this session's own read of `delta_star.jl`/`equilibrium.jl` surfaced) --
#
# `theta_free` (length `2*D^2-2`, `:logf` parameterization, `delta_star.jl`'s
# `expand_free_theta`) is:
#
#   theta_free[1]                    = log(gamma_prime_j)
#   theta_free[2 : 1+nA]              = A_free   (nA = D^2-1, one physical cell PER entry,
#                                                  in `ctx.A_pivot.other` order)
#   theta_free[2+nA : end]            = f_free_free (D^2-2 entries, one physical cell PER
#                                                  entry, in the f-pivot's `other` order)
#
# Three DISTINCT physical cells are reconstructed rather than packed directly, and are
# NEVER any coordinate's own "direct" cell:
#
#   - `A_pivot_cell` = `lin2od(ctx.A_pivot.pivot, D)`: `logA_full[pivot] = -sum_k
#     c[other[k]]*A_free[k] / c[pivot]` (g0=0 always for the A-pivot) -- a DENSE linear
#     combination of EVERY A_free coordinate. Always off-diagonal (`build_gravity_pivots`
#     restricts candidacy to `o!=d`), so `A_pivot_cell != (j,j)`.
#   - `f_pivot_cell`: same mechanism over `f_free_free`, but its OFFSET `g0_f =
#     c_full[jj_lin]*log(f_jj)` itself depends on `f_jj`, which depends on `gamma_prime_j`
#     AND `A[j,j]` -- so `f_pivot_cell` is ALSO affected by coordinate 1 and by whichever
#     coordinate packs `A[j,j]` directly, IN ADDITION to every f_free_free coordinate.
#     THIS DEPENDENCY IS NOT NAMED in the immediately-prior session's own Section 6 sketch
#     (which only lists `(j,j)` and the focal-link column for coordinate 1) -- an omission
#     this session's own derivation catches before any code is written on top of it. Always
#     distinct from `A_pivot_cell` and from `(j,j)` (`f_gravity_pivot_avoid_indices`
#     excludes both from f-pivot candidacy).
#   - `(j,j)` itself: `f[j,j]` is DERIVED (`derive_fjj_from_autarky_cutoff`), never packed
#     or pivoted -- affected by coordinate 1 and by whichever coordinate packs `A[j,j]`
#     directly (which always exists as an ordinary A_free coordinate, since the A-pivot can
#     never land on a diagonal cell).
#
# WHETHER a coordinate's affected cells reach the focal-link column (`profit_j`, which sums
# realized operating profit over ALL `D` destinations from origin `j`) is DATA-DEPENDENT,
# not a fixed structural fact independent of the fixture: a coordinate touches the link
# column iff ANY of its affected trade cells has origin `j` -- which can happen not only via
# its own direct cell, but via `A_pivot_cell`/`f_pivot_cell` themselves landing on an
# origin-`j` (export) cell. This is checked from the ACTUAL pivot cells at `ctx` construction
# time (via `melitz_pivot_map`), not assumed either way.

"""
    MelitzPivotMap

Precomputed, THETA-INDEPENDENT physical-cell identities for the three reconstructed cells
(`A_pivot`, `f_pivot`, `jj`) and every free coordinate's own direct physical cell(s) --
built once per `ctx` (the pivot CHOICE depends only on `ctx.c_full`/`ctx.A_pivot`/
`ctx.f_free_lin`, never on `theta`, `gravity_pivot_cells`'s own docstring: "WHICH cell is
chosen depends only on `c` and the avoid-set, never on `g0`").
"""
struct MelitzPivotMap
    D::Int
    j::Int
    nA::Int              # = D^2-1, length of the A_free block
    jj_cell::Tuple{Int,Int}
    A_pivot_cell::Tuple{Int,Int}
    f_pivot_cell::Tuple{Int,Int}
    A_direct_cell::Vector{Tuple{Int,Int}}   # length nA: A_free[idx]'s own physical cell
    f_direct_cell::Vector{Tuple{Int,Int}}   # length D^2-2: f_free_free[idx]'s own physical cell
    A_jj_index::Int      # which A_free index packs A[j,j] directly (always exists)
end

function melitz_pivot_map(ctx)
    D, j = ctx.D, ctx.target_country
    nA = D^2 - 1
    jj_cell = (j, j)
    A_pivot_cell = lin2od(ctx.A_pivot.pivot, D)
    A_direct_cell = [lin2od(i, D) for i in ctx.A_pivot.other]

    avoid_f = f_gravity_pivot_avoid_indices(D, ctx.f_free_lin, ctx.A_pivot.pivot)
    # g0 does not affect WHICH cell is chosen (only its reconstructed value) -- 0.0 is a
    # placeholder, matching `gravity_pivot_cells`'s own use of this exact call for the same
    # reason.
    f_pivot = build_gravity_pivot(ctx.c_full[ctx.f_free_lin], 0.0; avoid=avoid_f)
    f_pivot_cell = lin2od(ctx.f_free_lin[f_pivot.pivot], D)
    f_direct_cell = [lin2od(ctx.f_free_lin[i], D) for i in f_pivot.other]

    A_jj_index = findfirst(==(jj_cell), A_direct_cell)
    A_jj_index === nothing && error(
        "melitz_pivot_map: A[j,j] not found among the A-pivot's own direct cells -- " *
        "the A-pivot's diagonal-avoid invariant (build_gravity_pivots) must have changed")

    return MelitzPivotMap(D, j, nA, jj_cell, A_pivot_cell, f_pivot_cell,
        A_direct_cell, f_direct_cell, A_jj_index)
end

"""
    MelitzCoordDependency

One free coordinate's claimed affected set: `cells` (physical `(o,d)` trade pairs whose
moment column this coordinate can move) and `touches_link` (whether the focal-link column
can move). A SUPERSET claim (Section 7.1's own validation gate checks this), not
necessarily the tightest possible set.
"""
struct MelitzCoordDependency
    cells::Vector{Tuple{Int,Int}}
    touches_link::Bool
end

"""
    melitz_localized_dependency_map(ctx) -> Vector{MelitzCoordDependency}

Main prompt Section 7.1: for every one of the `2*D^2-2` free coordinates, the claimed
affected physical trade cells and whether the focal-link column is touched -- built PURELY
from `melitz_pivot_map(ctx)`'s theta-independent structure, no gradient/finite-difference
logic. See this file's header for the full derivation; summary per coordinate:

  - coordinate 1 (`gamma_prime_j`): `{jj_cell, f_pivot_cell}`, `touches_link=true` always
    (gamma enters the autarky firm's `price_power_autarky` directly, independent of `f_jj`).
  - an A_free coordinate (physical cell `c`): `{c, A_pivot_cell}`, PLUS `{jj_cell,
    f_pivot_cell}` if `c == jj_cell` (i.e. this is the coordinate packing `A[j,j]`).
    `touches_link` iff `c[1]==j` or `A_pivot_cell[1]==j` (or `c==jj_cell`, itself `o==j`).
  - an f_free_free coordinate (physical cell `c`): `{c, f_pivot_cell}`.
    `touches_link` iff `c[1]==j` or `f_pivot_cell[1]==j`.
"""
function melitz_localized_dependency_map(ctx)
    pm = melitz_pivot_map(ctx)
    j = pm.j
    deps = Vector{MelitzCoordDependency}(undef, 1 + pm.nA + length(pm.f_direct_cell))

    deps[1] = MelitzCoordDependency([pm.jj_cell, pm.f_pivot_cell], true)

    for idx in 1:pm.nA
        c = pm.A_direct_cell[idx]
        cells = Tuple{Int,Int}[c, pm.A_pivot_cell]
        touches_link = c[1] == j || pm.A_pivot_cell[1] == j
        if idx == pm.A_jj_index
            push!(cells, pm.jj_cell, pm.f_pivot_cell)
            touches_link = true
        end
        deps[1+idx] = MelitzCoordDependency(unique(cells), touches_link)
    end

    off = 1 + pm.nA
    for idx in eachindex(pm.f_direct_cell)
        c = pm.f_direct_cell[idx]
        cells = Tuple{Int,Int}[c, pm.f_pivot_cell]
        touches_link = c[1] == j || pm.f_pivot_cell[1] == j
        deps[off+idx] = MelitzCoordDependency(unique(cells), touches_link)
    end

    return deps
end

# ============================================================================
# GATE 2: the column-restricted `:method_b_localized` Jacobian backend, built ONLY on top
# of Gate 1's now-validated dependency map (main prompt Section 7.2's own required order:
# "a THIN wrapper that still calls fixed_active_set_moments!/_fill_fixed_active_set_moments!
# but restricted to the affected columns only... NOT a hand-rolled incremental update yet").
# ============================================================================

"""
    make_melitz_moments_jacobian_b_localized(h) -> Function

Backend `:method_b_localized` (Phase II.11 Gate 2): identical calling convention and
central-difference construction to `make_melitz_moments_jacobian_b` (`finite_delta_outer.jl`),
but each coordinate's displaced moment build is restricted (`fixed_active_set_moments_restricted!`,
`gradient_lab.jl`) to ONLY that coordinate's own claimed affected columns
(`melitz_localized_dependency_map`), copying every OTHER column from a single base-point
`G(theta)` computed ONCE per gradient call (not once per coordinate). For an unaffected
column, both displaced buffers retain the IDENTICAL base value, so `G_jac[:,:,k]` for that
column is EXACTLY `0.0` for every draw -- not merely close to zero -- which is also what a
BIT-EXACT match against full `:method_b`'s own `(Gp_full-Gm_full)/(2h)` requires whenever
Gate 1's dependency-map claim for that coordinate/column pair is correct (see the test
suite's own "Phase II.11 Gate 2: bit-exact vs. full Method B" battery, the gate this backend
must clear before being trusted for any live outer solve).

The dependency map itself is built ONCE (cached in the closure, keyed by `ctx` object
identity) since it is theta-independent (`melitz_pivot_map`'s own docstring).
"""
function make_melitz_moments_jacobian_b_localized(h::Real)
    Gbase_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gp_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gm_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    profit_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    depmap_cache = Ref{Union{Nothing,Vector{MelitzCoordDependency}}}(nothing)
    ctx_cache = Ref{Any}(nothing)

    function melitz_moments_jacobian_b_localized!(K_jac, G_jac, theta, U, obj)
        n = length(theta)
        K_jac .= 0.0
        K_jac[:, 1] .= 1.0
        if size(U, 1) < size(obj.U, 1)
            G_jac .= 0.0
            return nothing
        end
        ctx = obj.γ
        W = size(obj.U, 1)
        d = ctx.moment_layout.num_moments
        if Gbase_buf[] === nothing || size(Gbase_buf[]) != (W, d)
            Gbase_buf[] = zeros(Float64, W, d)
            Gp_buf[] = zeros(Float64, W, d)
            Gm_buf[] = zeros(Float64, W, d)
            profit_buf[] = zeros(Float64, W)
        end
        if depmap_cache[] === nothing || ctx_cache[] !== ctx
            depmap_cache[] = melitz_localized_dependency_map(ctx)
            ctx_cache[] = ctx
        end
        Gbase, Gp, Gm, profit_j = Gbase_buf[], Gp_buf[], Gm_buf[], profit_buf[]
        depmap = depmap_cache[]

        # Base moments at theta itself (full, unrestricted) -- ONCE per gradient call, seeds
        # every coordinate's unaffected columns for both displacement buffers.
        fixed_active_set_moments!(Gbase, profit_j, theta, ctx, obj)

        ei = zeros(n)
        @inbounds for k in 1:n
            dep = depmap[k]
            ei[k] = 1.0
            copyto!(Gp, Gbase)
            copyto!(Gm, Gbase)
            fixed_active_set_moments_restricted!(Gp, profit_j, theta .+ h .* ei, ctx, obj;
                cells=dep.cells, compute_link=dep.touches_link)
            fixed_active_set_moments_restricted!(Gm, profit_j, theta .- h .* ei, ctx, obj;
                cells=dep.cells, compute_link=dep.touches_link)
            @views G_jac[:, :, k] .= (Gp .- Gm) ./ (2h)
            ei[k] = 0.0
        end
        return nothing
    end
    return melitz_moments_jacobian_b_localized!
end

# ============================================================================
# Phase II.12 (this continuation session): parallel coordinate sweep on top of the now-
# validated, bit-exact serial localized backend (Section 11 above). Main prompt Section 3's
# own required design:
#
#   - the base inner solve (which produces `obj`'s dual `x`/`H` state this gradient reads
#     via `dual_scalar_at_fixed_G`/PsiObjectiveBundle's own functor machinery) must have
#     ALREADY completed before any coordinate work starts -- true by construction here: this
#     function is called as `moments_jacobian!` from `cc_algo`'s own `calculate_grad_k!`,
#     which is only ever invoked AFTER the current outer point's inner solve; nothing in
#     this function itself launches or waits on an inner solve;
#   - the base state (`Gbase`, `theta`, `ctx`, `obj`) is read-only for the whole parallel
#     region -- ONLY `Gbase` is shared, and no thread ever writes to it, only `copyto!`s FROM
#     it into its own private buffer;
#   - one thread-local scratch object per Julia thread (`Gp_bufs[]`/`Gm_bufs[]`/
#     `profit_bufs[]`, each a `Vector` of length `Threads.nthreads()`, indexed by
#     `Threads.threadid()` under `:static` scheduling -- the same iteration-to-thread
#     mapping this repo's own `docs/fullA_inner_blas_threading_report.md`/
#     `full_aod_diag/d4_exact/bench_threading.jl` Context B convention relies on) -- never a
#     single shared `Gp_buf`/`Gm_buf` (that would be the exact bug this session's own governing
#     prompt warns against, "no shared mutable moment columns");
#   - `G_jac[:, :, k]` for distinct `k` are disjoint views of the SAME output array -- safe
#     for concurrent writes with no synchronization needed;
#   - no inner KNITRO solve can be launched from inside a coordinate thread
#     (`fixed_active_set_moments_restricted!` is pure economic algebra over a FIXED dual
#     `x_base`/moment matrix, never touching KNITRO) -- guarded, not just assumed, via
#     `cc_algo/parallelism_guards.jl`'s existing `guard_enter_coord_pool!`/
#     `guard_exit_coord_pool!` (the SAME mutual-exclusion invariant checker
#     `inner_loop_KNITRO` already calls into on the Ricardian/fullA side), reused here rather
#     than inventing a second guard mechanism, matching this repo's own documented prior
#     regression (`fullA_nested_knitro_solve_hang_fixed.md`) from omitting exactly this class
#     of guard;
#   - BLAS threads forced to 1 for the duration of the coordinate sweep (`melitz_firm`'s own
#     per-draw scalar arithmetic does not call BLAS, but the main prompt's Section 3
#     requirement is unconditional -- honored here regardless), and the caller's own prior
#     BLAS thread count is restored afterward via `try/finally` even if a coordinate throws;
#   - no global RNG mutation anywhere in this call (none of the functions on this path touch
#     `Random`).
"""
    make_melitz_moments_jacobian_b_localized_parallel(h) -> Function

Backend `:B_localized_parallel` (Phase II.12): identical calling convention, dependency map,
and central-difference construction to `:method_b_localized` (`make_melitz_moments_jacobian_b_localized`
above), but the `n` coordinate probes run under `Threads.@threads :static` instead of a
serial `for` loop. Requires numerical equivalence -- in fact BIT-IDENTITY, not merely
`isapprox` -- to the serial localized backend, since each coordinate's own output column
`G_jac[:,:,k]` is computed from the identical inputs (`Gbase`, `theta`, `h`) regardless of
which thread executes it, with no floating-point-order-dependent reduction across threads.
Validated in the test suite ("Phase II.12: parallel localized gradient").
"""
function make_melitz_moments_jacobian_b_localized_parallel(h::Real)
    Gbase_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gbase_profit_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    depmap_cache = Ref{Union{Nothing,Vector{MelitzCoordDependency}}}(nothing)
    ctx_cache = Ref{Any}(nothing)
    Gp_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    Gm_bufs = Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    profit_bufs = Ref{Union{Nothing,Vector{Vector{Float64}}}}(nothing)
    nthreads_alloc = Ref(0)

    function melitz_moments_jacobian_b_localized_parallel!(K_jac, G_jac, theta, U, obj)
        n = length(theta)
        K_jac .= 0.0
        K_jac[:, 1] .= 1.0
        if size(U, 1) < size(obj.U, 1)
            G_jac .= 0.0
            return nothing
        end
        ctx = obj.γ
        W = size(obj.U, 1)
        d = ctx.moment_layout.num_moments
        # `Threads.threadid()` ranges over `1:Threads.maxthreadid()`, NOT `1:Threads.nthreads()`
        # -- Julia's `:default`/`:interactive` threadpool split (1.9+) means the global thread id
        # a task lands on can exceed the `:default`-pool count `Threads.nthreads()` returns. Sizing
        # these per-thread buffers by `Threads.nthreads()` alone throws a live `BoundsError` the
        # moment a task is scheduled onto a thread outside that range (found live this session at
        # `JULIA_NUM_THREADS=2` during the thread-count sweep) -- `Threads.maxthreadid()` is the
        # correct, safe upper bound for indexing by `Threads.threadid()`.
        nt = Threads.maxthreadid()

        if Gbase_buf[] === nothing || size(Gbase_buf[]) != (W, d)
            Gbase_buf[] = zeros(Float64, W, d)
            Gbase_profit_buf[] = zeros(Float64, W)
        end
        if Gp_bufs[] === nothing || nthreads_alloc[] != nt || size(Gp_bufs[][1]) != (W, d)
            Gp_bufs[] = [zeros(Float64, W, d) for _ in 1:nt]
            Gm_bufs[] = [zeros(Float64, W, d) for _ in 1:nt]
            profit_bufs[] = [zeros(Float64, W) for _ in 1:nt]
            nthreads_alloc[] = nt
        end
        if depmap_cache[] === nothing || ctx_cache[] !== ctx
            depmap_cache[] = melitz_localized_dependency_map(ctx)
            ctx_cache[] = ctx
        end
        Gbase, Gbase_profit = Gbase_buf[], Gbase_profit_buf[]
        depmap = depmap_cache[]

        # Base moments at theta itself -- ONCE, SERIALLY, before any coordinate thread starts
        # (the base state must be fully built and immutable for the parallel region below).
        fixed_active_set_moments!(Gbase, Gbase_profit, theta, ctx, obj)

        prev_blas_threads = BLAS.get_num_threads()
        guards_on = isdefined(Main, :CounterfactualSensitivity)
        BLAS.set_num_threads(1)
        guards_on && Main.CounterfactualSensitivity.guard_enter_coord_pool!()
        try
            Threads.@threads :static for k in 1:n
                tid = Threads.threadid()
                Gp, Gm, profit_j = Gp_bufs[][tid], Gm_bufs[][tid], profit_bufs[][tid]
                dep = depmap[k]
                copyto!(Gp, Gbase)
                copyto!(Gm, Gbase)
                theta_p = copy(theta)
                theta_p[k] += h
                theta_m = copy(theta)
                theta_m[k] -= h
                fixed_active_set_moments_restricted!(Gp, profit_j, theta_p, ctx, obj;
                    cells=dep.cells, compute_link=dep.touches_link)
                fixed_active_set_moments_restricted!(Gm, profit_j, theta_m, ctx, obj;
                    cells=dep.cells, compute_link=dep.touches_link)
                @views G_jac[:, :, k] .= (Gp .- Gm) ./ (2h)
            end
        finally
            guards_on && Main.CounterfactualSensitivity.guard_exit_coord_pool!()
            BLAS.set_num_threads(prev_blas_threads)
        end
        return nothing
    end
    return melitz_moments_jacobian_b_localized_parallel!
end
