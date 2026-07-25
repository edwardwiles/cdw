# Sorted-tail moment-construction optimization (2026-07-25 session). See
# docs/melitz_sorted_tail_optimization_2026-07-25.md for the full mathematical derivation,
# validation, and benchmark results.
#
# CORE OBSERVATION (report Section A): for a fixed origin o, bilateral participation
# `1{z_so >= cutoff_od}` is a scalar threshold test on the SAME z_so across every
# destination d (firm_quantities.jl's `melitz_firm`: `active = operating_profit > 0`,
# strictly monotone increasing in z since price is strictly decreasing in z and revenue is
# strictly increasing in price^(1-sigma) for sigma>1). Sorting z[:,o] ONCE per origin lets
# every destination's active set be found by one binary search instead of a full W-row
# scan, without changing which draws participate or their contribution -- a pure
# algebraic/data-layout optimization, not an approximation.
#
# CRITICAL: only origin-LOCAL sorting is used. `permutation[:,o] = sortperm(z[:,o])` is
# stored per origin; every active-tail computation recovers the ORIGINAL row index via this
# permutation before writing into G, so joint draws (row s pairs (z_s1,...,z_sD) together,
# the QMC dependence structure) are never decoupled across origins. It is NOT permissible to
# independently sort every origin and pair the k-th sorted observation across origins as if
# it were one new joint draw.

"""
    MelitzSortedTailContext

Immutable per-origin sorted view of the reference draw matrix `z` (`W x D`), built ONCE per
fixed draw context and reused across every FC/GA callback at that context. Never mutated,
never re-sorted, after construction.

- `z_original`, `log_z_original`, `z_power_original` (`z^(sigma-1)`): the untouched
  original `W x D` draws in three representations, indexed by the ORIGINAL row `s`.
- `permutation[:,o] = sortperm(z_original[:,o])`, so `z_original[permutation[:,o],o]` is
  nondecreasing.
- `sorted_z`, `sorted_log_z`, `sorted_z_power`: the SAME three representations, reindexed
  into origin `o`'s own sorted order (`sorted_z[:,o] = z_original[permutation[:,o],o]`).
- `fingerprint`: a content hash of `(D, W, sigma, theta_star (if given), hash(z_draws))` --
  a caller comparing fingerprints can detect a stale context reused with changed draws, `W`,
  or `sigma` before it silently produces wrong output.

The permutation is ORIGIN-LOCAL. `permutation[:,o]` reorders origin `o`'s OWN draw column
only -- it must NEVER be applied to a different origin's column, and sorted position `k`
must NEVER be treated as "the same joint draw" across two origins. The cross-origin
dependence structure of the original QMC draw matrix lives entirely in the shared row index
`s`, which every active-tail computation recovers via `permutation[:,o]` before writing back
into a `W`-indexed output (a scatter-add/scatter-write, never a reordered output).
"""
struct MelitzSortedTailContext
    D::Int
    W::Int
    sigma::Float64
    z_original::Matrix{Float64}
    log_z_original::Matrix{Float64}
    z_power_original::Matrix{Float64}
    permutation::Matrix{Int}
    sorted_z::Matrix{Float64}
    sorted_log_z::Matrix{Float64}
    sorted_z_power::Matrix{Float64}
    fingerprint::UInt
end

"""
    build_melitz_sorted_tail_context(z_draws, sigma; theta_star=nothing) -> MelitzSortedTailContext

Builds the immutable sorted-tail context once from the `W x D` reference draw matrix.
`theta_star`, if supplied, is folded into the fingerprint only (the sort/power computations
below use `sigma` alone -- `z^(sigma-1)` does not depend on the generating Pareto shape);
passing it gives a caller with `theta_star` on hand extra staleness protection, matching the
governing prompt's own suggested fingerprint field list.
"""
function build_melitz_sorted_tail_context(z_draws::AbstractMatrix, sigma::Real;
                                           theta_star::Union{Nothing,Real}=nothing)
    W, D = size(z_draws)
    sigma_f = Float64(sigma)
    z_original = Matrix{Float64}(z_draws)
    log_z_original = log.(z_original)
    z_power_original = z_original .^ (sigma_f - 1)

    permutation = Matrix{Int}(undef, W, D)
    sorted_z = Matrix{Float64}(undef, W, D)
    sorted_log_z = Matrix{Float64}(undef, W, D)
    sorted_z_power = Matrix{Float64}(undef, W, D)
    @inbounds for o in 1:D
        perm_o = sortperm(@view z_original[:, o])
        permutation[:, o] .= perm_o
        sorted_z[:, o] .= @view z_original[perm_o, o]
        sorted_log_z[:, o] .= @view log_z_original[perm_o, o]
        sorted_z_power[:, o] .= @view z_power_original[perm_o, o]
    end

    h = hash(:melitz_sorted_tail_ctx_v1)
    h = hash(D, h)
    h = hash(W, h)
    h = hash(sigma_f, h)
    theta_star !== nothing && (h = hash(Float64(theta_star), h))
    h = hash(z_original, h)

    return MelitzSortedTailContext(D, W, sigma_f, z_original, log_z_original, z_power_original,
                                    permutation, sorted_z, sorted_log_z, sorted_z_power, UInt(h))
end

"""
    melitz_active_tail_start(sorted_col, cutoff) -> k

First 1-based sorted position `k` with `sorted_col[k] > cutoff` (STRICT -- matches the
production active convention, `melitz_firm`'s `active = operating_profit > 0`, i.e. a draw
exactly AT the cutoff is INACTIVE, probability zero under the continuous reference Pareto
but exercised deliberately by the near-tie tests). Positions `k:W` are active, `1:k-1`
inactive. Returns `length(sorted_col)+1` if nothing is active (cutoff at/above every draw);
`1` if everything is active (cutoff below every draw). `sorted_col` must already be sorted
ascending (the caller's `MelitzSortedTailContext` invariant guarantees this -- not
re-checked here, for speed, since this runs once per (o,d) cell per callback).
"""
function melitz_active_tail_start(sorted_col::AbstractVector, cutoff::Real)
    isnan(cutoff) && throw(ArgumentError("melitz_active_tail_start: cutoff is NaN"))
    cutoff == Inf && return length(sorted_col) + 1
    cutoff == -Inf && return 1
    return searchsortedlast(sorted_col, cutoff) + 1
end

"""
    melitz_moments_sorted_tail!(K, G, p, eq, cf, sorted_ctx, layout; X_data=eq.trade_flow)

`:sorted_tail_serial` backend for `melitz_moments!` (moments.jl) -- numerically equivalent
(tight agreement; not bit-identical, since the arithmetic is reassociated around
precomputed `z^(sigma-1)`/`coef_od` rather than recomputed per-draw through
`melitz_firm`/`unconstrained_revenue`) trade-share block, computed via `sorted_ctx`'s
per-origin sorted draws and ONE binary search per `(o,d)` cell instead of a full `W`-row
scan and per-draw firm evaluation.

The single focal link column (`layout.focal_link_index`) is computed DENSELY in this
backend too (Phase 3.1 audit, docs Section C.1: it DOES have the same scalar cutoff-tail
structure in `z[:,target_country]`, but is `O(D*W)` versus the trade-share block's
`O(D^2*W)` -- asymptotically negligible at `D=20` -- so this session did not optimize it,
keeping the extra autarky/`gamma_prime_target` algebra out of the validated fast path).

Requires `sorted_ctx.D == p.D`, `sorted_ctx.W == size(G,1)`, `sorted_ctx.sigma == p.sigma`
(the sorted `z^(sigma-1)` transform is sigma-specific) -- mismatches throw `ArgumentError`
rather than silently reusing stale sorted state.
"""
function melitz_moments_sorted_tail!(K::AbstractVector, G::AbstractMatrix, p::MelitzPrimitives,
                                      eq::MelitzEquilibrium, cf::MelitzCounterfactual,
                                      sorted_ctx::MelitzSortedTailContext, layout::MelitzMomentLayout;
                                      X_data::AbstractMatrix=eq.trade_flow)
    D = p.D
    W = sorted_ctx.W
    D == sorted_ctx.D || throw(ArgumentError(
        "melitz_moments_sorted_tail!: sorted_ctx.D=$(sorted_ctx.D) != p.D=$D"))
    p.sigma == sorted_ctx.sigma || throw(ArgumentError(
        "melitz_moments_sorted_tail!: sorted_ctx.sigma=$(sorted_ctx.sigma) != p.sigma=$(p.sigma) " *
        "-- stale sorted context (built under a different sigma)"))
    size(G, 1) == W || throw(ArgumentError("G must have sorted_ctx.W rows"))
    size(G, 2) == layout.num_moments || throw(ArgumentError("G must have layout.num_moments columns"))
    length(K) == W || throw(ArgumentError("K must have length W"))

    sigma = p.sigma
    j = p.target_country

    @melitz_profile :moments_trade_share_sorted begin
        @inbounds for o in 1:D
            perm_o = @view sorted_ctx.permutation[:, o]
            zpow_o = @view sorted_ctx.sorted_z_power[:, o]
            sorted_z_o = @view sorted_ctx.sorted_z[:, o]
            for d in 1:D
                trade_col = layout.trade_index[o, d]
                lambda_od = X_data[o, d] / eq.expenditure[d]
                @views G[:, trade_col] .= -lambda_od
                cutoff_od = eq.cutoff[o, d]
                C_od = melitz_C(p.w[o], p.tau[o, d], p.A[o, d], sigma, eq.expenditure[d])
                coef_od = C_od / eq.expenditure[d]
                k = melitz_active_tail_start(sorted_z_o, cutoff_od)
                for pos in k:W
                    s = perm_o[pos]
                    G[s, trade_col] = coef_od * zpow_o[pos] - lambda_od
                end
            end
        end
    end

    profit_j = zeros(eltype(G), W)
    z_orig = sorted_ctx.z_original
    @melitz_profile :moments_focal_link_sorted begin
        @inbounds for d in 1:D
            for w in 1:W
                z = z_orig[w, j]
                firm = melitz_firm(p.w[j], p.tau[j, d], p.A[j, d], p.f[j, d], sigma,
                                    eq.expenditure[d], 1.0, z)
                profit_j[w] += firm.realized_operating_profit
            end
        end
        link_col = layout.focal_link_index
        price_power_autarky = p.gamma_prime_target
        @inbounds for w in 1:W
            z_j = z_orig[w, j]
            firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], sigma,
                                        cf.expenditure_prime, price_power_autarky, z_j)
            G[w, link_col] = profit_j[w] / p.w[j] - firm_autarky.realized_operating_profit / cf.w_prime
        end
    end

    K .= p.gamma_prime_target - 1
    return nothing
end

"""
    melitz_moments_sorted_tail_parallel!(K, G, p, eq, cf, sorted_ctx, layout; X_data=eq.trade_flow)

`:sorted_tail_parallel` backend: identical to `melitz_moments_sorted_tail!` except the outer
`o in 1:D` loop over the trade-share block runs on `Threads.@threads`. This is safe WITHOUT
any additional synchronization because `layout.trade_index[o,d]` is a BIJECTION onto
`1:D^2` -- for a FIXED `o`, the `D` columns `trade_index[o,1:D]` are entirely disjoint from
every other origin `o'`'s own `D` columns, so concurrent threads write disjoint columns of
`G` and never race. Each thread only reads its own origin's slice of `sorted_ctx` (also
disjoint per column of the `W x D` sorted matrices) and writes thread-local scalars
(`lambda_od`, `C_od`, `coef_od`, `k`) -- no shared mutable scratch. The focal link column
(computed after the parallel region, over `d in 1:D` at the single fixed `target_country`)
remains serial -- it is a single column, `O(D*W)`, not worth parallelizing at this scale,
and accumulates into a single shared `profit_j` vector that a naive origin-parallel loop
would race on.

Callers running this backend under KNITRO must set `BLAS.set_num_threads(1)` for the
duration (this repo's own standing convention for any Julia-thread-parallel region,
`feedback-openblas-threads-hard-cap-violation` memory) -- not enforced here, since this
function has no BLAS calls of its own, but the CALLING driver's overall thread budget must
still respect it.
"""
function melitz_moments_sorted_tail_parallel!(K::AbstractVector, G::AbstractMatrix, p::MelitzPrimitives,
                                               eq::MelitzEquilibrium, cf::MelitzCounterfactual,
                                               sorted_ctx::MelitzSortedTailContext, layout::MelitzMomentLayout;
                                               X_data::AbstractMatrix=eq.trade_flow)
    D = p.D
    W = sorted_ctx.W
    D == sorted_ctx.D || throw(ArgumentError(
        "melitz_moments_sorted_tail_parallel!: sorted_ctx.D=$(sorted_ctx.D) != p.D=$D"))
    p.sigma == sorted_ctx.sigma || throw(ArgumentError(
        "melitz_moments_sorted_tail_parallel!: sorted_ctx.sigma=$(sorted_ctx.sigma) != p.sigma=$(p.sigma) " *
        "-- stale sorted context (built under a different sigma)"))
    size(G, 1) == W || throw(ArgumentError("G must have sorted_ctx.W rows"))
    size(G, 2) == layout.num_moments || throw(ArgumentError("G must have layout.num_moments columns"))
    length(K) == W || throw(ArgumentError("K must have length W"))

    sigma = p.sigma
    j = p.target_country

    @melitz_profile :moments_trade_share_sorted_parallel begin
        Threads.@threads for o in 1:D
            perm_o = @view sorted_ctx.permutation[:, o]
            zpow_o = @view sorted_ctx.sorted_z_power[:, o]
            sorted_z_o = @view sorted_ctx.sorted_z[:, o]
            @inbounds for d in 1:D
                trade_col = layout.trade_index[o, d]
                lambda_od = X_data[o, d] / eq.expenditure[d]
                @views G[:, trade_col] .= -lambda_od
                cutoff_od = eq.cutoff[o, d]
                C_od = melitz_C(p.w[o], p.tau[o, d], p.A[o, d], sigma, eq.expenditure[d])
                coef_od = C_od / eq.expenditure[d]
                k = melitz_active_tail_start(sorted_z_o, cutoff_od)
                for pos in k:W
                    s = perm_o[pos]
                    G[s, trade_col] = coef_od * zpow_o[pos] - lambda_od
                end
            end
        end
    end

    profit_j = zeros(eltype(G), W)
    z_orig = sorted_ctx.z_original
    @melitz_profile :moments_focal_link_sorted begin
        @inbounds for d in 1:D
            for w in 1:W
                z = z_orig[w, j]
                firm = melitz_firm(p.w[j], p.tau[j, d], p.A[j, d], p.f[j, d], sigma,
                                    eq.expenditure[d], 1.0, z)
                profit_j[w] += firm.realized_operating_profit
            end
        end
        link_col = layout.focal_link_index
        price_power_autarky = p.gamma_prime_target
        @inbounds for w in 1:W
            z_j = z_orig[w, j]
            firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], sigma,
                                        cf.expenditure_prime, price_power_autarky, z_j)
            G[w, link_col] = profit_j[w] / p.w[j] - firm_autarky.realized_operating_profit / cf.w_prime
        end
    end

    K .= p.gamma_prime_target - 1
    return nothing
end

"""
    melitz_sorted_tail_diagnostics(eq, sorted_ctx, layout) -> NamedTuple

Phase 5: fused cell-level participation diagnostics computed via ONE binary search per
`(o,d)` cell (`O(D^2 log W)`) instead of a full `W`-row Boolean-mask scan
(`min_active_draw_count`/`cell_participation_diagnostics`, moments.jl, `O(D^2 W)`). Returns
`active_count[o,d] = W-k_od+1`, `active_fraction[o,d]`, `first_active_z[o,d]` (the smallest
active draw for that cell, `NaN` if none active), `max_z[o,d]` (`sorted_ctx`'s own maximum
draw for origin `o`, broadcast across `d` since it does not depend on the destination).
Validated against `min_active_draw_count`/`cell_participation_diagnostics`'s dense output
(test suite).
"""
function melitz_sorted_tail_diagnostics(eq::MelitzEquilibrium, sorted_ctx::MelitzSortedTailContext,
                                         layout::MelitzMomentLayout)
    D = sorted_ctx.D
    W = sorted_ctx.W
    active_count = zeros(Int, D, D)
    active_fraction = zeros(Float64, D, D)
    first_active_z = fill(NaN, D, D)
    max_z_o = zeros(Float64, D)
    @inbounds for o in 1:D
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        max_z_o[o] = sorted_z_o[end]
        for d in 1:D
            cutoff_od = eq.cutoff[o, d]
            k = melitz_active_tail_start(sorted_z_o, cutoff_od)
            n_active = W - k + 1
            active_count[o, d] = n_active
            active_fraction[o, d] = n_active / W
            n_active > 0 && (first_active_z[o, d] = sorted_z_o[k])
        end
    end
    max_z = [max_z_o[o] for o in 1:D, d in 1:D]
    return (active_count=active_count, active_fraction=active_fraction,
            first_active_z=first_active_z, max_z=max_z)
end
