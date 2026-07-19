# ============================================================================
# Continuation 9, Phase 5: closed-form (non-bisection) bandwidth selection.
#
# `select_bandwidth` (composite_gradient.jl) finds h via geometric bisection
# over `mass_at(h)`, up to `max_iter`+1 = 7 O(W) `count_winner_flips` probes
# per A-block coordinate -- measured as the single largest piece of the full
# 400-coordinate `L_fix` gradient at D=20/W=80000
# (docs/fullA_D20_W80k_microbenchmark.md sec 3D: 15.7s of a ~20.5s
# reconstructed gradient total).
#
# This file replaces the bisection with a DIRECT, EXACT quantile lookup for
# the common case, derived from the price formula itself rather than searched
# for numerically:
#
#   price_and_pTsigma_cell shows price(o,d) = constCons_od(θ) / U[:,o].^(-μ),
#   and constCons_od(θ) = wHat[o]*τ[o,d] * AodPow(θ), AodPow = (lvl/cHat)^(-μ),
#   lvl = aod_level_cell(θ,o,d) = Aod_θ[o,d] * (DATA CONSTANTS, independent of
#   Aod_θ). So, for FIXED (o,d,ω), price is proportional to Aod_θ[o,d]^(-μ).
#
#   In the pivot-reduced coordinates (gravity_elimination.jl), perturbing
#   `w[coord_idx]` by `h` moves z[dir_lin] = z0[dir_lin] + h (slope EXACTLY 1)
#   and z[piv_lin] = z0[piv_lin] + slope_piv*h (slope_piv = -c[dir_lin]/c[piv_lin],
#   a CONSTANT, from pivot_expand's affine formula) -- i.e. Aod_θ[cell](h) =
#   Aod_θ0[cell] * exp(slope_cell * h) exactly (no small-h approximation).
#
#   Combining: price[ω](h) = price0[ω] * exp(c*h), c := -μ*slope_cell, a
#   SINGLE SCALAR (same sign/magnitude for every draw ω) -- i.e. every draw's
#   price at this cell moves monotonically in the SAME direction as h grows.
#   update_winner_o1's own case analysis (lfix_incremental.jl) says exactly
#   when a flip occurs for a single changed origin o at destination d:
#     - o is currently the winner (winner0[ω,d]==o): flips iff new_price >
#       runnerup_price0[ω,d] -- only reachable if c>0 (price rising).
#     - o is not currently the winner: flips iff new_price < winner_price0[ω,d]
#       -- only reachable if c<0 (price falling).
#   Both are ONE inequality in a KNOWN monotonic function of h, so the exact
#   crossing h_flip[ω] solves in closed form:
#     h_flip = log(threshold_price / price0[ω]) / c
#   No bisection needed -- an O(W) pass computes every draw's exact crossing
#   point directly, then the target switching-mass fraction is read off as an
#   ORDER STATISTIC of the (sorted) h_flip array, not searched for.
#
# SCOPE / what this does NOT close-form: coordinates whose two affected cells
# (dir_lin, piv_lin) fall in the SAME destination (both origins move at once
# -- exactly the D-1-of-D^2-1 coordinates in the pivot's own destination
# column) need the full top-3 case analysis for TWO simultaneously-moving
# competitors, which has more crossing sub-cases (either mover could win,
# against a moving OR a fixed threshold) -- not derived here given this task's
# time budget. `select_bandwidth_quantile` detects this case explicitly and
# falls back to the proven `select_bandwidth` bisection for exactly those
# coordinates (~D-1 of D^2-1, ~4.5% at D=20) -- correctness preserved
# everywhere, the closed-form fast path only claimed where it was actually
# derived. Reported honestly in docs/fullA_D20_bandwidth_optimization_report.md
# rather than silently degrading accuracy on the collision coordinates.
# ============================================================================
include(joinpath(@__DIR__, "composite_gradient.jl"))   # -> select_bandwidth (fallback), affected_cells, LFixBaseCache

"""
    coord_cell_slopes(pe, coord_idx) -> (dir_lin, slope_dir, piv_lin, slope_piv)

The exact affine coefficients of z[dir_lin] and z[piv_lin] w.r.t. w[coord_idx]
implied by `pivot_expand` (gravity_elimination.jl): `slope_dir` is always
`1.0` (the direct free coordinate enters its own z entry unchanged);
`slope_piv = -pe.c[dir_lin]/pe.c[piv_lin]` is the pivot's compensating affine
slope that keeps `pivot_expand`'s output exactly gravity-feasible for every
`w[coord_idx]`. Pure algebra read off `pivot_expand`'s own formula, not a new
derivation.
"""
function coord_cell_slopes(pe, coord_idx::Int)
    k = coord_idx - 1
    dir_lin = pe.other_idx[k]
    piv_lin = pe.pivot_lin
    slope_piv = -pe.c[dir_lin] / pe.c[piv_lin]
    return dir_lin, 1.0, piv_lin, slope_piv
end

"""
    exact_flip_thresholds!(hflips, cache, μ, o, d, c) -> Int

Appends every FINITE, POSITIVE exact crossing h for cell (o,d) under slope
constant `c = -μ*slope_cell` to `hflips` (a caller-owned, pre-sized buffer --
appended via `push!`, kept a plain `Vector{Float64}` since the candidate count
per cell is at most W and this is called at most twice per coordinate).
Returns the number of finite crossings found (diagnostic only). See this
file's header for the derivation; this is the O(1)-per-draw evaluation of
that closed form, no bisection.
"""
function exact_flip_thresholds!(hflips::Vector{Float64}, cache::LFixBaseCache, μ::Float64, o::Int, d::Int, c::Float64)
    n_found = 0
    W = cache.W
    if c > 0
        # only currently-winning draws can flip (winner's price rises past runner-up)
        @inbounds for ω in 1:W
            cache.winner0[ω, d] == o || continue
            p0 = cache.winner_price0[ω, d]
            thr = cache.runnerup_price0[ω, d]
            hf = log(thr / p0) / c
            if hf >= 0.0 && isfinite(hf)
                push!(hflips, hf); n_found += 1
            end
        end
    elseif c < 0
        # only currently-losing draws can flip (o's price falls past the current winner)
        @inbounds for ω in 1:W
            cache.winner0[ω, d] == o && continue
            p0 = cache.price0[ω, o, d]
            thr = cache.winner_price0[ω, d]
            hf = log(thr / p0) / c
            if hf >= 0.0 && isfinite(hf)
                push!(hflips, hf); n_found += 1
            end
        end
    end
    # c == 0: price constant in h at this cell, no draw ever flips from this cell alone
    return n_found
end

"""
    select_bandwidth_quantile(cache, ctx, pe, w0, coord_idx; kwargs...) -> (h, switch_mass, meta)

Drop-in ALTERNATIVE to `select_bandwidth` (same signature/return contract),
computing the adaptive bandwidth via the direct closed-form quantile lookup
described in this file's header instead of geometric bisection. Falls back to
`select_bandwidth` unchanged for the same-destination-two-changed-origins
case (see header "SCOPE"). `max_iter` is accepted for interface compatibility
(passed through to the fallback) but unused on the closed-form path (no
iteration needed there).
"""
function select_bandwidth_quantile(cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int;
        h0::Float64 = 0.01, h_floor::Float64 = 1e-4, h_ceil::Float64 = 0.1,
        target_mass_frac::Tuple{Float64,Float64} = (0.003, 0.03), max_iter::Int = 6,
        multi_method::Symbol = :top3)

    cells = affected_cells(pe, coord_idx)
    @assert !isempty(cells) "select_bandwidth_quantile: coord_idx=$coord_idx has no affected A_od cells"
    dests = last.(cells)

    if length(unique(dests)) < length(cells)
        # same-destination collision (2 changed origins, 1 destination) -- not
        # closed-formed here, fall back to the proven bisection selector.
        h, m, meta = select_bandwidth(cache, ctx, pe, w0, coord_idx; h0 = h0, h_floor = h_floor, h_ceil = h_ceil,
                                        target_mass_frac = target_mass_frac, max_iter = max_iter, multi_method = multi_method)
        return h, m, merge(meta, (method = :bisection_fallback_same_dest,))
    end

    dir_lin, slope_dir, piv_lin, slope_piv = coord_cell_slopes(pe, coord_idx)
    μ = cache.μ
    W = cache.W
    n_dests = length(cells)   # always 2 here (the collision case returned above)

    hflips = Float64[]
    sizehint!(hflips, 2 * W)
    (o1, d1) = cells[1]; c1 = -μ * slope_dir
    (o2, d2) = cells[2]; c2 = -μ * slope_piv
    exact_flip_thresholds!(hflips, cache, μ, o1, d1, c1)
    exact_flip_thresholds!(hflips, cache, μ, o2, d2, c2)

    N = W * n_dests
    lo_frac, hi_frac = target_mass_frac
    if isempty(hflips)
        # this cell's price never crosses ANY threshold under a +h probe (both
        # c1,c2 have the "wrong" sign for every draw, or a genuine degenerate
        # cell) -- no mass achievable at any h; report h_ceil (matches
        # select_bandwidth's own "hit_ceil" terminal state when mass never
        # reaches lo_frac), mass 0.
        return h_ceil, 0.0, (n_iter = 0, hit_floor = false, hit_ceil = true, method = :quantile, n_candidates = 0)
    end
    sort!(hflips)
    # Target the FLOOR of the band, not its midpoint: `select_bandwidth`'s bisection
    # only needs to clear `lo_frac` and stops as soon as it does (its geometric
    # doubling/halving searches for ANY h in-band, not the band's center) --
    # empirically it lands close to `lo_frac`, not `(lo_frac+hi_frac)/2` (verified
    # in test_bandwidth_quantile.jl: mass_bisect medians sit near the low end of
    # [0.003,0.03] across a real coordinate sweep). Matching that target directly
    # (rather than the band's midpoint) is what makes the quantile pick agree with
    # the bisection's actual choice instead of systematically over-shooting to a
    # much larger h -- an early version of this file targeted the midpoint and
    # produced gradients with A-block cosine similarity ~0.58 to the bisection
    # baseline (see report), corrected here, not silently smoothed over.
    rank = clamp(ceil(Int, lo_frac * N), 1, length(hflips))
    h_raw = hflips[rank]
    h = clamp(h_raw, h_floor, h_ceil)
    m = count(x -> x <= h, hflips) / N

    return h, m, (n_iter = 0, hit_floor = h == h_floor, hit_ceil = h == h_ceil, method = :quantile,
                  n_candidates = length(hflips), h_raw = h_raw)
end
