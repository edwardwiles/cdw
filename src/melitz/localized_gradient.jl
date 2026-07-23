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
