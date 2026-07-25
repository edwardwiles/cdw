# Phase I.3 (screening-session continuation): exact compressed origin-block feasibility
# screen. Exploits the one-dimensional Melitz structure (all of an origin's D bilateral
# cutoffs -- and, for the focal origin, its extra autarky cutoff -- are thresholds on the
# SAME per-draw productivity axis z[:,o]) rather than the full W-row convex-hull LP.
#
# -- Derivation --
#
# For origin o, destination d: `G[w,trade_col] = c[o,d]*y_w*1{z_w>=zhat[o,d]} - lambda[o,d]`
# where `y_w = z[w,o]^(sigma-1)`, `c[o,d] = C_od/expenditure[d]` (`C_od = melitz_C(w[o],
# tau[o,d],A[o,d],sigma,expenditure[d])`, `firm_quantities.jl`), `lambda[o,d] =
# X_data[o,d]/expenditure[d]`. The moment condition `E_p[G[:,trade_col]]=0` for a probability
# vector `p` over the `W` draws is therefore `E_p[y*1{active}] = H[o,d] := lambda[o,d]/c[o,d]`
# -- a REQUIRED TAIL MOMENT on the same y=z^(sigma-1) transform, gated by the SAME per-draw
# threshold `zhat[o,d]` that varies only by destination.
#
# Sorting origin o's D cutoffs (plus, for the focal origin, its autarky cutoff -- see below)
# ascending partitions the z axis into intervals within which the ACTIVE SET of destinations
# is constant. Let `m_k` = probability mass placed in interval `k`, `T_k` = the p-weighted
# y-moment `E_p[y*1{interval k}]` within interval `k`. Since the SAMPLE min/max of `y` within
# interval `k` (over the `W` draws actually falling there) bound what `T_k` can be for ANY
# mass `m_k` (`ymin_k*m_k <= T_k <= ymax_k*m_k`), and destination `d`'s equation only
# involves the SUM of `T_k` over intervals at or above its own cutoff rank, the origin's own
# D trade-share moments (jointly) are feasible iff there exist `m_k>=0` (`sum m_k = 1`) and
# `T_k` in the stated box satisfying every destination's linear equation -- an LP with
# `O(D)` variables/constraints, NOT `O(W)`. This is EXACT (necessary AND sufficient) for this
# origin's own trade-share moments considered alone: any feasible `(m_k,T_k)` is realizable
# by an actual probability vector using at most 2 draws per interval (the argmin/argmax
# attaining draws, mixed to hit `T_k` exactly) -- the SAME two-point-mixture argument the
# range screen's file header uses for a single column, extended interval-by-interval. It is
# a NECESSARY (not sufficient) condition for the FULL D^2+1-column joint system, since it
# ignores every OTHER origin's columns.
#
# For the FOCAL origin (`o == ctx.target_country`), the autarky firm's zero-profit cutoff is
# an EXTRA breakpoint on the SAME z_j axis (a "virtual (D+1)-th destination"), and the
# focal-link moment `G[:,focal_link_index]` is an EXTRA equation combining every baseline
# destination's OPERATING PROFIT (not revenue -- affine in y, `(C_od/sigma)*y - w_j*f[j,d]`
# when active) against the autarky operating profit, divided by `w[j]`/`w_prime`
# respectively. Both are affine in `(T_k,m_k)`, so this extra row is folded into the SAME LP
# for the focal origin only.

using JuMP, HiGHS

"""
    melitz_origin_intervals(o, theta, ctx, obj) -> (breakpoints, rank, autarky_rank)

Sorted, DEDUPLICATED cutoff breakpoints for origin `o`'s productivity axis: the `D`
bilateral cutoffs `zhat[o,:]`, PLUS (only if `o == ctx.target_country`) the autarky
zero-profit cutoff. `rank[d]` is the 1-based index into `breakpoints` of destination `d`'s
own cutoff (destination `d` is active throughout every interval `k >= rank[d]`).
`autarky_rank` is `nothing` unless `o == ctx.target_country`.
"""
function melitz_origin_intervals(o::Int, theta::AbstractVector, ctx, obj)
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    D = ctx.D
    zhat = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    cuts = zhat[o, :]   # length D, cuts[d] = zhat[o,d]

    autarky_cut = nothing
    j = ctx.target_country
    if o == j
        expenditure_prime = ctx.w_prime * ctx.L[j]
        C_jj_auk = melitz_C(ctx.w_prime, 1.0, A[j, j], ctx.sigma, expenditure_prime)
        autarky_cut = melitz_cutoff(ctx.w_prime, f_jj * gamma_prime_j, ctx.sigma, C_jj_auk)
    end

    all_cuts = autarky_cut === nothing ? copy(cuts) : vcat(cuts, autarky_cut)
    breakpoints = sort(unique(all_cuts))
    rank = [searchsortedfirst(breakpoints, cuts[d]) for d in 1:D]
    autarky_rank = autarky_cut === nothing ? nothing : searchsortedfirst(breakpoints, autarky_cut)
    return (breakpoints=breakpoints, rank=rank, autarky_rank=autarky_rank,
            A=A, f=f, gamma_prime_j=gamma_prime_j, f_jj=f_jj, zhat=zhat)
end

"""
    melitz_origin_block_lp(o, theta, ctx, obj; optimizer=HiGHS.Optimizer) -> Bool

Builds and solves the Phase I.3 compressed origin-block feasibility LP for origin `o`.
Returns `true` iff the LP is feasible (no certificate); `false` iff INFEASIBLE (a valid
`InfiniteDeltaCertified` certificate for this origin's own trade-share moments -- and, for
the focal origin, the focal-link moment too).
"""
function melitz_origin_block_lp(o::Int, theta::AbstractVector, ctx, obj; optimizer=HiGHS.Optimizer)
    D = ctx.D
    W = size(obj.U, 1)
    iv = melitz_origin_intervals(o, theta, ctx, obj)
    breakpoints, rank, autarky_rank = iv.breakpoints, iv.rank, iv.autarky_rank
    K = length(breakpoints)   # intervals 1..K are "at or above breakpoint k"; interval 0 is below all

    z = @view obj.U[:, o]
    y = similar(z, Float64)
    @inbounds for w in 1:W
        y[w] = z[w]^(ctx.sigma - 1)
    end

    # bucket draws into intervals 0..K: interval k (1<=k<=K) is [breakpoints[k], breakpoints[k+1))
    # (breakpoints[K+1] = +Inf); interval 0 is z < breakpoints[1].
    interval_of = [searchsortedlast(breakpoints, zv) for zv in z]   # 0..K (0 = below all breakpoints)
    ymin = fill(Inf, K); ymax = fill(-Inf, K); has_draw = falses(K)
    @inbounds for w in 1:W
        k = interval_of[w]
        k == 0 && continue
        has_draw[k] = true
        y[w] < ymin[k] && (ymin[k] = y[w])
        y[w] > ymax[k] && (ymax[k] = y[w])
    end

    model = Model(optimizer)
    set_silent(model)
    @variable(model, m[0:K] >= 0)
    @variable(model, T[1:K])
    @constraint(model, sum(m[k] for k in 0:K) == 1)
    for k in 1:K
        if has_draw[k]
            @constraint(model, T[k] >= ymin[k] * m[k])
            @constraint(model, T[k] <= ymax[k] * m[k])
        else
            # no draw at all falls in this interval -- no mass can be placed there, and its
            # (vacuous) y-moment is pinned to zero.
            @constraint(model, m[k] == 0)
            @constraint(model, T[k] == 0)
        end
    end

    C = [melitz_C(ctx.w[o], ctx.tau[o, d], iv.A[o, d], ctx.sigma, ctx.expenditure[d]) for d in 1:D]
    c_od = [C[d] / ctx.expenditure[d] for d in 1:D]
    lambda_od = [ctx.X_data[o, d] / ctx.expenditure[d] for d in 1:D]
    H = lambda_od ./ c_od
    for d in 1:D
        @constraint(model, sum(T[k] for k in rank[d]:K) == H[d])
    end

    j = ctx.target_country
    if o == j && autarky_rank !== nothing
        expenditure_prime = ctx.w_prime * ctx.L[j]
        C_jj_auk = melitz_C(ctx.w_prime, 1.0, iv.A[j, j], ctx.sigma, expenditure_prime)
        w_j = ctx.w[j]
        link_expr = @expression(model,
            sum(sum((C[d] / ctx.sigma) * T[k] - w_j * iv.f[j, d] * m[k] for k in rank[d]:K) for d in 1:D) / w_j -
            sum((C_jj_auk / (ctx.sigma * iv.gamma_prime_j)) * T[k] - ctx.w_prime * iv.f_jj * m[k] for k in autarky_rank:K) / ctx.w_prime)
        @constraint(model, link_expr == 0)
    end

    optimize!(model)
    st = termination_status(model)
    return st == JuMP.OPTIMAL || st == JuMP.FEASIBLE_POINT
end

"""
    melitz_origin_block_screen(theta, ctx, obj; optimizer=HiGHS.Optimizer) -> Union{Nothing,InfiniteDeltaCertified}

Runs `melitz_origin_block_lp` for every origin `o=1..D`; returns the FIRST origin's
infeasibility as an `InfiniteDeltaCertified(o, NaN, NaN, :origin_block)` certificate (the
`lo`/`hi` fields are not meaningful for this multi-column certificate -- `column` records
the ORIGIN, not a single moment column), or `nothing` if every origin's block LP is
feasible.
"""
function melitz_origin_block_screen(theta::AbstractVector, ctx, obj; optimizer=HiGHS.Optimizer)
    D = ctx.D
    for o in 1:D
        if !melitz_origin_block_lp(o, theta, ctx, obj; optimizer=optimizer)
            return InfiniteDeltaCertified(o, NaN, NaN, :origin_block)
        end
    end
    return nothing
end

"""
    melitz_origin_block_lp_reference(o, theta, ctx, obj; optimizer=HiGHS.Optimizer) -> Bool

VALIDATION ONLY (main prompt Section 3's own requirement: "build a generic origin-block
convex-hull LP using all W rows as a trusted reference") -- NOT for production use (`O(W)`
variables, one `p[w]` per draw, vs. the compressed screen's `O(D)`). Encodes the IDENTICAL
constraint set as `melitz_origin_block_lp` (the same D trade-share equations, plus the
focal-link equation for `o == ctx.target_country`) directly over the raw draws, with no
interval compression -- mathematically the SAME feasible region (the compressed LP is an
exact reformulation, not a relaxation: any `(m_k,T_k)` satisfying the compressed LP is
realized by a `p` supported on at most 2 draws per interval, which reproduces the identical
`(m_k,T_k)` and therefore satisfies every constraint here too), so the two should agree
EXACTLY on every point, not merely "reference never contradicts a compressed rejection."
"""
function melitz_origin_block_lp_reference(o::Int, theta::AbstractVector, ctx, obj; optimizer=HiGHS.Optimizer)
    D = ctx.D
    W = size(obj.U, 1)
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    zhat = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    z = @view obj.U[:, o]
    y = z .^ (ctx.sigma - 1)

    model = Model(optimizer)
    set_silent(model)
    @variable(model, p[1:W] >= 0)
    @constraint(model, sum(p) == 1)

    C = [melitz_C(ctx.w[o], ctx.tau[o, d], A[o, d], ctx.sigma, ctx.expenditure[d]) for d in 1:D]
    c_od = [C[d] / ctx.expenditure[d] for d in 1:D]
    lambda_od = [ctx.X_data[o, d] / ctx.expenditure[d] for d in 1:D]
    H = lambda_od ./ c_od
    for d in 1:D
        active = zhat[o, d]
        @constraint(model, sum(p[w] * y[w] for w in 1:W if z[w] >= active) == H[d])
    end

    j = ctx.target_country
    if o == j
        expenditure_prime = ctx.w_prime * ctx.L[j]
        C_jj_auk = melitz_C(ctx.w_prime, 1.0, A[j, j], ctx.sigma, expenditure_prime)
        auk_cut = melitz_cutoff(ctx.w_prime, f_jj * gamma_prime_j, ctx.sigma, C_jj_auk)
        w_j = ctx.w[j]
        link_expr = @expression(model,
            sum(sum((C[d] / ctx.sigma) * p[w] * y[w] - w_j * f[j, d] * p[w] for w in 1:W if z[w] >= zhat[j, d]) for d in 1:D) / w_j -
            sum((C_jj_auk / (ctx.sigma * gamma_prime_j)) * p[w] * y[w] - ctx.w_prime * f_jj * p[w] for w in 1:W if z[w] >= auk_cut) / ctx.w_prime)
        @constraint(model, link_expr == 0)
    end

    optimize!(model)
    st = termination_status(model)
    return st == JuMP.OPTIMAL || st == JuMP.FEASIBLE_POINT
end

"""
    melitz_origin_block_monotonicity_check(o, theta, ctx, obj) -> Bool

The main prompt's MINIMUM required check (Section 3, "at minimum, verify the necessary
monotonicity"): for origin `o`, if `zhat[o,a] <= zhat[o,b]` then `H[o,a] >= H[o,b]` (a
lower cutoff's tail moment must be at least as large as a higher cutoff's, since the lower
cutoff's active set is a SUPERSET with the same nonnegative `y` integrand). Returns `true`
if the monotonicity holds for every pair; a `false` is itself a cheap `InfiniteDeltaCertified`
certificate (implied by, but far cheaper than, the full LP above).
"""
function melitz_origin_block_monotonicity_check(o::Int, theta::AbstractVector, ctx, obj)
    D = ctx.D
    A, f, _, _ = melitz_expand_theta(theta, ctx)
    zhat = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    C = [melitz_C(ctx.w[o], ctx.tau[o, d], A[o, d], ctx.sigma, ctx.expenditure[d]) for d in 1:D]
    c_od = [C[d] / ctx.expenditure[d] for d in 1:D]
    lambda_od = [ctx.X_data[o, d] / ctx.expenditure[d] for d in 1:D]
    H = lambda_od ./ c_od
    for a in 1:D, b in 1:D
        if zhat[o, a] <= zhat[o, b] && H[a] < H[b] - 1e-10
            return false
        end
    end
    return true
end
