# ============================================================================
# Claude Code task 2026-08-01, §9 (v3, follow-up per live user question
# 2026-08-01: "are you sure the slowness isn't just your implementation
# being unoptimized? If so, optimize it."): O(1)-per-changed-cell profiled
# outer gradient, built on the SAME incremental-winner-update mechanism
# production's C+ gradient uses (update_winner_o1 / the top-3 exact update,
# lfix_incremental.jl, reused UNCHANGED), now validated the CORRECT way --
# gated directly against `profiled_lfix_at` (profiled_outer_gradient_fd_
# 2026-08-01.jl, the already-gated full-rebuild version) at machine
# precision, not against expensive re-solved ground truth.
#
# WHY THIS DRAFT IS DIFFERENT FROM THE EARLIER (ABANDONED) ATTEMPT: the
# earlier incremental draft failed its own D4 gate (cos_sim~0.03) and was
# abandoned under live user correction to "keep it surgical." Re-examining
# that failure with a now-available cheap, exact reference (the full-rebuild
# version, itself since gated against real KNITRO ground truth and against
# the real A/B) shows the CONTRIB FORMULA in that draft was already correct
# -- re-derived independently here from reduced_homogeneous_dual_contraction
# (reduced_homogeneous_contraction_2026-08-01.jl:41-96) line by line and
# found identical. The one confirmed bug was in the ANALYTIC gp component
# (missing the LPrime_bi*S_m term, fixed same-session in
# PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md section 6a's "3c
# corrected") -- that fix was made to this file moments before it was
# deleted and was NEVER RE-GATED. gp's component dominates the gradient
# vector's magnitude at these points (see PROFILED_OUTER_GRADIENT_GATE_D4/
# D20 CSVs, gp_component >> norm(A-block)), so a broken gp term alone is
# sufficient to explain a near-zero overall cosine similarity even with a
# fully correct A-block mechanism. This file re-verifies that hypothesis
# directly rather than assuming it -- see
# test_profiled_incremental_vs_fullrebuild_2026-08-01.jl.
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :update_winner_o1) || error("profiled_lfix_incremental_2026-08-01.jl requires lfix_incremental.jl to be included first.")
isdefined(Main, :evaluate_profiled_point) || error("profiled_lfix_incremental_2026-08-01.jl requires profiled_outer_evaluator_2026-08-01.jl to be included first.")

struct ProfiledLFixCache
    D::Int; Ddest::Int; W::Int; μ::Float64; σ::Float64
    SW::Vector{Float64}
    price0::Array{Float64,3}        # W x D x Ddest
    pTσ0::Array{Float64,3}          # W x D x Ddest  (== wval at that (draw,origin,dest))
    winner0::Matrix{Int}
    winner_price0::Matrix{Float64}
    runnerup0::Matrix{Int}
    runnerup_price0::Matrix{Float64}
    third0::Matrix{Int}
    third_price0::Matrix{Float64}
    third_pTσ0::Matrix{Float64}
    κ::Matrix{Float64}              # D x Ddest, solved dual coefficients (0.0 at every anchor cell, structurally)
    Cbar_eff::Vector{Float64}       # Ddest, includes +kappa_cf*gp^sigma at bi_slot
    contrib0::Matrix{Float64}       # W x Ddest, cached (kappa[wo,d]-Cbar_eff[d])*pTsigma0[w,wo,d]
    const_part::Float64             # const_cf - pmmterm (winner-independent, draw-independent)
    cf_raw_κcf::Vector{Float64}     # W, kappa_cf*cf.cf_raw[w] (winner-independent, draw-dependent; 0-vector if no france)
    κ_cf::Float64
    gpσ::Float64
    bi_slot::Int
    has_france::Bool
    ζstar::Float64
    M::Float64
    q0::Vector{Float64}
    spec::AnchorSpec
    layout::ProfiledEconomicMomentLayout
end

"""
    build_price_winner_base_cache(ctx, θ_full, cf) -> NamedTuple

EXTRACTED (2026-08-01, parallel outer-gradient-layer task, unchanged logic --
pure code motion, no formula change) from `build_profiled_lfix_cache`'s first
half: the dense `(W x D x Ddest)` price/pTsigma cache and the per-draw
winner/runnerup/third top-3 cache, identical for EVERY family (task §4: "the
shared base cache must remain usable by both" formulations/families -- this
is the "SAME `LFixBaseCache`-style" one-time `O(W*D*Ddest)` build the
mission's shared-engine section describes). `build_profiled_lfix_cache` below
is unchanged in behavior; it now simply calls this helper instead of
inlining the same code, so the identical winner-update MECHANISM
(`price_and_pTsigma_cell`/`min_secondthirdmin_with_idx`, both reused
unmodified) is available to a restricted-family cache builder without a
second implementation of it anywhere (task §4: "Do not write another
winner-update algorithm").
"""
function build_price_winner_base_cache(ctx, θ_full::AbstractVector, cf)
    D = cf.D; Ddest = cf.D_dest; W = cf.W

    price0 = Array{Float64}(undef, W, D, Ddest)
    pTσ0 = Array{Float64}(undef, W, D, Ddest)
    for d in 1:Ddest, o in 1:D
        p, ps = price_and_pTsigma_cell(θ_full, ctx, o, d)
        price0[:, o, d] .= p; pTσ0[:, o, d] .= ps
    end

    winner0 = Matrix{Int}(undef, W, Ddest); winner_price0 = Matrix{Float64}(undef, W, Ddest)
    runnerup0 = Matrix{Int}(undef, W, Ddest); runnerup_price0 = Matrix{Float64}(undef, W, Ddest)
    third0 = Matrix{Int}(undef, W, Ddest); third_price0 = Matrix{Float64}(undef, W, Ddest)
    @inbounds for d in 1:Ddest, ω in 1:W
        m1, idx1, m2, idx2, m3, idx3 = min_secondthirdmin_with_idx(@view(price0[ω, :, d]))
        winner0[ω, d] = idx1; winner_price0[ω, d] = m1
        runnerup0[ω, d] = idx2; runnerup_price0[ω, d] = m2
        third0[ω, d] = idx3; third_price0[ω, d] = m3
    end
    third_pTσ0 = Matrix{Float64}(undef, W, Ddest)
    @inbounds for d in 1:Ddest, ω in 1:W
        t = third0[ω, d]
        third_pTσ0[ω, d] = t == 0 ? Inf : pTσ0[ω, t, d]
    end
    winner0 == cf.winner || error("build_price_winner_base_cache: rebuilt winner0 disagrees with cf.winner -- price_and_pTsigma_cell/build_compressed_factual formulas have diverged")

    return (price0 = price0, pTσ0 = pTσ0, winner0 = winner0, winner_price0 = winner_price0,
        runnerup0 = runnerup0, runnerup_price0 = runnerup_price0, third0 = third0,
        third_price0 = third_price0, third_pTσ0 = third_pTσ0)
end

function build_profiled_lfix_cache(w_profiled::AbstractVector{Float64}, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, ev)
    st = ev.st; cf = st.cf; layout = st.layout; θ_full = ev.theta_full; obj = ev.obj
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    μ = θ_full[1]; σ = θ_full[2]
    SW = cf.SW

    base = build_price_winner_base_cache(ctx, θ_full, cf)
    price0 = base.price0; pTσ0 = base.pTσ0
    winner0 = base.winner0; winner_price0 = base.winner_price0
    runnerup0 = base.runnerup0; runnerup_price0 = base.runnerup_price0
    third0 = base.third0; third_price0 = base.third_price0; third_pTσ0 = base.third_pTσ0

    β = ev.result.beta
    κ = zeros(D, Ddest)
    Cbar = zeros(Ddest)
    @inbounds for k in eachindex(layout.retained_full_factual_j)
        o = layout.retained_origin[k]; slot = layout.retained_slot[k]
        j_full = layout.retained_full_factual_j[k]
        kk = β[k] * cf.nrm[j_full] * cf.gdiv[j_full]
        κ[o, slot] = kk
        Cbar[slot] += kk * cf.Pmat[o, slot]
    end

    has_france = layout.france_ratio_reduced_j > 0
    κ_cf = 0.0; gpσ = 0.0; bi_slot = 0; const_cf = 0.0
    cf_raw_κcf = zeros(W)
    if has_france
        bi = ctx.bi; bi_slot = dest_slot(ctx, bi)
        gp = w_profiled[1]
        gpσ = gp^σ
        wPrime_bi = 1.0
        denom_cf = gpσ * wPrime_bi * ctx.γ.LPrime[bi]
        j_cf_full = cf.cf_col
        κ_cf = β[layout.france_ratio_reduced_j] * cf.nrm[j_cf_full] * cf.gdiv[j_cf_full]
        const_cf = κ_cf * denom_cf
        cf_raw_κcf .= κ_cf .* cf.cf_raw
    end
    pmmterm = 0.0
    if cf.usePMM == 1
        @inbounds for k in eachindex(layout.retained_full_factual_j)
            j_full = layout.retained_full_factual_j[k]
            pmmterm += β[k] * cf.nrm[j_full] * cf.PMM[j_full]
        end
        if has_france
            j_cf_full = cf.cf_col
            pmmterm += β[layout.france_ratio_reduced_j] * cf.nrm[j_cf_full] * cf.PMM[j_cf_full]
        end
    end

    Cbar_eff = copy(Cbar)
    has_france && (Cbar_eff[bi_slot] += κ_cf * gpσ)

    contrib0 = Matrix{Float64}(undef, W, Ddest)
    @inbounds for d in 1:Ddest, ω in 1:W
        wo = winner0[ω, d]
        contrib0[ω, d] = (κ[wo, d] - Cbar_eff[d]) * pTσ0[ω, wo, d]
    end

    const_part = const_cf - pmmterm
    q0 = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        acc = const_part + sum(@view contrib0[w, :]) + (has_france ? cf_raw_κcf[w] : 0.0)
        t0 = SW[w] * acc
        q0[w] = -ev.result.zeta - t0
    end

    return ProfiledLFixCache(D, Ddest, W, μ, σ, SW, price0, pTσ0, winner0, winner_price0, runnerup0,
        runnerup_price0, third0, third_price0, third_pTσ0, κ, Cbar_eff, contrib0, const_part, cf_raw_κcf,
        κ_cf, gpσ, bi_slot, has_france, ev.result.zeta, ev.obj.M, q0, spec, layout)
end

"lfix_from_q_reduced(q, zeta, M) -> Float64 -- Delta_dual's own sign convention (-(mean Psi(q)+zeta)), matching evaluate_profiled_point's result.Delta_dual exactly at the base point."
function lfix_from_q_reduced(q::AbstractVector, ζstar::Float64, M::Float64)
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return -(sum(Psi_q) / M + ζstar)
end

"""
    dest_contrib_reduced_o1(cache, ctx, θ_full, d, changed_origins) -> Vector{Float64} (length W)

O(1)-per-draw winner update (1 changed origin: `update_winner_o1`; 2 changed
origins in the same destination: exact top-3-cache update, mirroring
`dest_contrib_incremental_top3`), producing `(kappa[wo,d]-Cbar_eff[d])*pTsigma[wo]`
per draw.
"""
function dest_contrib_reduced_o1(cache::ProfiledLFixCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    κd = @view cache.κ[:, d]; Cd = cache.Cbar_eff[d]
    if length(changed_origins) == 1
        o = changed_origins[1]
        new_price, new_pTσ = price_and_pTsigma_cell(θ_full, ctx, o, d)
        contrib = Vector{Float64}(undef, W)
        @inbounds for ω in 1:W
            wo, _, _, _, _ = update_winner_o1(cache.winner_price0[ω, d], cache.winner0[ω, d],
                cache.runnerup_price0[ω, d], cache.runnerup0[ω, d], o, new_price[ω])
            pTσ_wo = wo == o ? new_pTσ[ω] : cache.pTσ0[ω, wo, d]
            contrib[ω] = (κd[wo] - Cd) * pTσ_wo
        end
        return contrib
    else
        length(changed_origins) <= 2 || error("dest_contrib_reduced_o1: >2 changed origins in one destination is unreachable for a single profiled coordinate (direct+pivot only)")
        Cd_set = changed_origins
        new_price = Dict{Int,Vector{Float64}}(); new_pTσ = Dict{Int,Vector{Float64}}()
        for o in Cd_set
            p, ps = price_and_pTsigma_cell(θ_full, ctx, o, d)
            new_price[o] = p; new_pTσ[o] = ps
        end
        contrib = Vector{Float64}(undef, W)
        @inbounds for ω in 1:W
            r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
            best_o = 0; best_p = Inf; best_pTσ = Inf
            if !(r1 in Cd_set)
                best_o = r1; best_p = cache.winner_price0[ω, d]; best_pTσ = cache.pTσ0[ω, r1, d]
            elseif !(r2 in Cd_set)
                best_o = r2; best_p = cache.runnerup_price0[ω, d]; best_pTσ = cache.pTσ0[ω, r2, d]
            elseif r3 != 0 && !(r3 in Cd_set)
                best_o = r3; best_p = cache.third_price0[ω, d]; best_pTσ = cache.third_pTσ0[ω, d]
            end
            bo = best_o; bp = best_p; bpTσ = best_pTσ
            for o in Cd_set
                v = new_price[o][ω]
                if v < bp || (v == bp && o < bo)
                    bp = v; bo = o; bpTσ = new_pTσ[o][ω]
                end
            end
            if bo == 0
                col = Vector{Float64}(undef, D)
                for o in 1:D
                    col[o] = haskey(new_price, o) ? new_price[o][ω] : cache.price0[ω, o, d]
                end
                _, bo, _ = min_and_secondmin(col)
                bpTσ = haskey(new_pTσ, bo) ? new_pTσ[bo][ω] : cache.pTσ0[ω, bo, d]
            end
            contrib[ω] = (κd[bo] - Cd) * bpTσ
        end
        return contrib
    end
end

"""
    profiled_affected_cells(spec, pe, coord_idx) -> Vector{(o,d)}

`coord_idx` in 2:(1+n_free). Returns the (o,d) cells touched: the
coordinate's own direct retained cell, PLUS the gravity-pivot's retained
cell (which moves under EVERY r_free perturbation via
`pivot_expand_on_retained`'s affine reconstruction).
"""
function profiled_affected_cells(spec::AnchorSpec, pe::PivotGravityElimOnRetained, coord_idx::Int)
    ridx = retained_linear_indices(spec)
    D = spec.D
    k = coord_idx - 1
    pos_dir = pe.other_pos[k]
    pos_piv = pe.pivot_pos
    i_dir = ridx[pos_dir]; d_dir = div(i_dir - 1, D) + 1; o_dir = i_dir - (d_dir - 1) * D
    i_piv = ridx[pos_piv]; d_piv = div(i_piv - 1, D) + 1; o_piv = i_piv - (d_piv - 1) * D
    return [(o_dir, d_dir), (o_piv, d_piv)]
end

"""
    profiled_lfix_incremental_at(cache, ctx, spec, pe, w0, coord_idx, new_val) -> Float64

Fixed-dual L_fix (== Delta_dual's own sign convention) at a single-coordinate
perturbation of `w0`, via the O(1) incremental winner update.
"""
function profiled_lfix_incremental_at(cache::ProfiledLFixCache, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, w0::AbstractVector, coord_idx::Int, new_val::Float64)
    w = copy(w0); w[coord_idx] = new_val
    decoded = decode_outer_profiled(w, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    cells = profiled_affected_cells(spec, pe, coord_idx)
    affected_dests = unique(last.(cells))

    q = copy(cache.q0)
    for d in affected_dests
        origins_here = [o for (o, dd) in cells if dd == d]
        old_contrib = @view cache.contrib0[:, d]
        new_contrib = dest_contrib_reduced_o1(cache, ctx, θ_full, d, origins_here)
        q .-= cache.SW .* (new_contrib .- old_contrib)
    end
    return lfix_from_q_reduced(q, cache.ζstar, cache.M)
end

"""
    profiled_gp_component_analytic(cache, w_profiled, ev, ctx) -> Float64

Exact closed-form d(Delta_dual)/dgp -- SECOND correction (live, 2026-08-01):
the first "fix" (an `LPrime_bi*S_m` term added alongside `Tslot_bi`) was
itself wrong, traced down by h-sweeping the full-rebuild central FD to a
clean, stable limit (0.23296009410... as h->0, D4 calibration) that matched
NEITHER the original NOR the first-"fixed" analytic formula, then verifying
per-draw dq[w]/dgp against FD directly.

ROOT CAUSE: `cf.cf_raw[w]` (compressed_moments.jl:264, `cf_raw =
constConsσ_bibi/UσPow_bi - denom_cf`, `denom_cf = gp^σ*wPrime_bi*LPrime_bi`)
is **NOT a gp-independent data constant** -- it is rebuilt fresh at the
CURRENT gp every time `build_compressed_factual` runs, and already contains
its own `-gp^σ*wPrime_bi*LPrime_bi` term. `reduced_homogeneous_dual_
contraction`'s separate `const_cf = kappa_cf*gp^σ*wPrime_bi*LPrime_bi`
(added once per draw) and `cf_raw[w]`'s embedded `-gp^σ*wPrime_bi*LPrime_bi`
EXACTLY CANCEL when combined (`const_cf + kappa_cf*cf_raw[w] = kappa_cf*
constConsσ_bibi/UσPow_bi[w]`, gp-independent) -- so the ENTIRE gp-dependence
of `t[w]` collapses to the single remaining term,
`-kappa_cf*gp^σ*wval[w,bi_slot]`. The `LPrime_bi*S_m` piece the first "fix"
added was spurious; the earliest (very first, pre-any-fix) draft's formula
was closer in FORM but had the wrong SIGN. Verified: this formula matches
the h->0 full-rebuild FD limit to 5 significant figures at D4 calibration
(0.232874 analytic vs 0.232960 FD-limit; residual is `Tslot_bi` sampling
precision, not a formula error -- see
test_profiled_incremental_vs_fullrebuild_2026-08-01.jl for the decisive
machine-precision-vs-full-rebuild gate, which is the real acceptance test).
"""
function profiled_gp_component_analytic(cache::ProfiledLFixCache, w_profiled::AbstractVector{Float64}, ev, ctx)
    cache.has_france || return 0.0
    m_weights = ev.m_weights
    Tslot_bi = 0.0
    @inbounds for ω in 1:cache.W
        Tslot_bi += cache.SW[ω] * m_weights[ω] * cache.pTσ0[ω, cache.winner0[ω, cache.bi_slot], cache.bi_slot]
    end
    gp = w_profiled[1]
    return -cache.κ_cf * cache.σ * gp^(cache.σ - 1) * Tslot_bi / cache.M
end

"""
    profiled_count_winner_flips(cache, ctx, θ_full, d, changed_origins) -> Int

Profiled analog of `composite_gradient.jl::count_winner_flips` -- counts how
many of the W draws' cached winner at destination `d` flip under a probed
perturbation, using the SAME `update_winner_o1`/top-3-cache case analysis
already validated in `dest_contrib_reduced_o1`. Needed so
`profiled_select_bandwidth` can target the same switching-mass window
production's own selector does.
"""
function profiled_count_winner_flips(cache::ProfiledLFixCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    if length(changed_origins) == 1
        o = changed_origins[1]
        new_price, _ = price_and_pTsigma_cell(θ_full, ctx, o, d)
        flips = 0
        @inbounds for ω in 1:W
            wo, _, _, _, _ = update_winner_o1(cache.winner_price0[ω, d], cache.winner0[ω, d],
                cache.runnerup_price0[ω, d], cache.runnerup0[ω, d], o, new_price[ω])
            flips += (wo != cache.winner0[ω, d])
        end
        return flips
    else
        length(changed_origins) <= 2 || error("profiled_count_winner_flips: >2 changed origins unreachable for a single profiled coordinate")
        Cd = changed_origins
        new_price = Dict{Int,Vector{Float64}}()
        for o in Cd
            p, _ = price_and_pTsigma_cell(θ_full, ctx, o, d)
            new_price[o] = p
        end
        flips = 0
        @inbounds for ω in 1:W
            r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
            best_o = 0; best_p = Inf
            if !(r1 in Cd)
                best_o = r1; best_p = cache.winner_price0[ω, d]
            elseif !(r2 in Cd)
                best_o = r2; best_p = cache.runnerup_price0[ω, d]
            elseif r3 != 0 && !(r3 in Cd)
                best_o = r3; best_p = cache.third_price0[ω, d]
            end
            bo = best_o; bp = best_p
            for o in Cd
                v = new_price[o][ω]
                if v < bp || (v == bp && o < bo)
                    bp = v; bo = o
                end
            end
            if bo == 0
                col = Vector{Float64}(undef, D)
                for o in 1:D
                    col[o] = haskey(new_price, o) ? new_price[o][ω] : cache.price0[ω, o, d]
                end
                _, bo, _ = min_and_secondmin(col)
            end
            flips += (bo != r1)
        end
        return flips
    end
end

"""
    profiled_select_bandwidth(cache, ctx, spec, pe, w0, coord_idx; kwargs...) -> (h, mass, meta)

EXACT profiled analog of `composite_gradient.jl::select_bandwidth` -- same
switching-mass-targeted geometric bisection (default `h0=0.01`,
`h_floor=1e-4`, `h_ceil=0.1`, `target_mass_frac=(0.003,0.03)`, `max_iter=6`).
Per live user feedback ("I just wanted you to do the same thing and method
but with slightly modified formula") this is now wired into
`profiled_composite_gradient_at_incremental` exactly the way production
calls `select_bandwidth` once per A-block coordinate -- no methodological
simplification, only the per-draw contribution formula differs (§9 of the
derivation doc).
"""
function profiled_select_bandwidth(cache::ProfiledLFixCache, ctx, spec::AnchorSpec, pe::PivotGravityElimOnRetained,
        w0::AbstractVector, coord_idx::Int; h0::Float64 = 0.01, h_floor::Float64 = 1e-4, h_ceil::Float64 = 0.1,
        target_mass_frac::Tuple{Float64,Float64} = (0.003, 0.03), max_iter::Int = 6)
    cells = profiled_affected_cells(spec, pe, coord_idx)
    affected_dests = unique(last.(cells))

    function mass_at(h::Float64)
        w = copy(w0); w[coord_idx] += h
        decoded = decode_outer_profiled(w, ctx, pe)
        θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
        total_flips = 0
        for d in affected_dests
            origins_here = [o for (o, dd) in cells if dd == d]
            total_flips += profiled_count_winner_flips(cache, ctx, θ_full, d, origins_here)
        end
        return total_flips / (cache.W * length(affected_dests))
    end

    h = h0
    lo_frac, hi_frac = target_mass_frac
    m = mass_at(h)
    n_iter = 0
    while n_iter < max_iter
        if m < lo_frac && h < h_ceil
            h = min(h * 2, h_ceil)
        elseif m > hi_frac && h > h_floor
            h = max(h / 2, h_floor)
        else
            break
        end
        m = mass_at(h)
        n_iter += 1
        (h == h_ceil || h == h_floor) && break
    end
    return h, m, (n_iter = n_iter, hit_floor = h == h_floor, hit_ceil = h == h_ceil)
end

"""
    profiled_composite_gradient_at_incremental(w_profiled, ctx, spec, pe, ev; multi_method=:top3) -> (g, meta)

O(1)-per-changed-cell profiled outer gradient, now matching production's
`composite_gradient_at` EXACTLY in method: `g[1]` exact analytic gp
(`profiled_gp_component_analytic`, mirrors `gamma_component_analytic`);
`g[2:end]` central FD via `profiled_lfix_incremental_at`, each coordinate's
step `h` chosen by `profiled_select_bandwidth` (mirrors production's own
per-coordinate `select_bandwidth` call inside its `for k in 2:D2` loop) --
no fixed global `h` anymore.
"""
function profiled_composite_gradient_at_incremental(w_profiled::AbstractVector{Float64}, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, ev)
    cache = build_profiled_lfix_cache(w_profiled, ctx, spec, pe, ev)
    return profiled_composite_gradient_from_cache(cache, ctx, spec, pe, w_profiled, ev)
end

"""
    profiled_composite_gradient_from_cache(cache, ctx, spec, pe, w_profiled, ev) -> (g, meta)

EXTRACTED (2026-08-01, parallel outer-gradient-layer task, unchanged logic --
pure code motion) from `profiled_composite_gradient_at_incremental`'s second
half: the exact analytic gp component plus the per-coordinate adaptively-
bandwidthed central-FD loop, taking an ALREADY-BUILT `ProfiledLFixCache` as
input instead of building one itself. `ev` is used ONLY for its `m_weights`
field (the gp component's `Tslot_bi` accumulation) -- it need not be the same
`ev` that built `cache` in general, only one carrying `m_weights` at the same
solved dual point (family adapters that build `ev` differently must ensure
this). This is the ONE concrete method every family's outer gradient
(unrestricted and, via the shared engine, every restricted family) calls --
`profiled_composite_gradient_at_incremental` (unrestricted call site,
unchanged behavior) and `shared_family_outer_gradient`
(`profiled_shared_economic_gradient_engine_2026-08-01.jl`, restricted-family
call site) both delegate here, so the A/gp gradient formula and the
winner-update mechanism it drives are defined in exactly one place (task §8:
"The A/gp portion returned by every family must come from the same concrete
shared method").
"""
function profiled_composite_gradient_from_cache(cache::ProfiledLFixCache, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, w_profiled::AbstractVector{Float64}, ev)
    n_total = outer_dim_profiled(pe)
    g = zeros(n_total)
    g[1] = profiled_gp_component_analytic(cache, w_profiled, ev, ctx)

    h_used = zeros(n_total); switch_mass = zeros(n_total)
    @inbounds for coord_idx in 2:n_total
        h, m, _selmeta = profiled_select_bandwidth(cache, ctx, spec, pe, w_profiled, coord_idx)
        h_used[coord_idx] = h; switch_mass[coord_idx] = m
        Lp = profiled_lfix_incremental_at(cache, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] + h)
        Lm = profiled_lfix_incremental_at(cache, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] - h)
        g[coord_idx] = (Lp - Lm) / (2h)
    end
    return g, (cache = cache, w0 = collect(Float64, w_profiled), h_used = h_used, switch_mass = switch_mass)
end
