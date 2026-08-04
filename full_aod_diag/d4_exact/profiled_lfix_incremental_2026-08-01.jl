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
isdefined(Main, :LFixFactorizedWorkspace) || error("profiled_lfix_incremental_2026-08-01.jl requires lfix_factorized_workspace.jl to be included first (Section 13 gradient-engine allocation fix, 2026-08-02: reuses FULL's own build_winner_ref!/pTσ_from_score instead of a separate dense price0/pTσ0 tensor).")

struct ProfiledLFixCache
    D::Int; Ddest::Int; W::Int; μ::Float64; σ::Float64
    SW::Vector{Float64}
    logCC0::Matrix{Float64}         # D x Ddest -- log(constCons0[o,d]), FULL's own compact score table (lfix_factorized_workspace.jl)
    mulU::Matrix{Float64}           # W x D -- μ*log(U[w,o]), FULL's own compact score table
    winner0::Matrix{Int}
    winner_price0::Matrix{Float64}  # now stores the WINNER's SCORE (log-price), not price -- see note below
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

# Performance closeout task (2026-08-02), Section 13 follow-up: the gradient-engine cost
# investigation (real timing evidence: ~6.5-9s/call at W=20,000, dominated by allocation of fresh
# W x D x Ddest price0/pTσ0 arrays on EVERY gradient call) found that FULL's OWN production
# gradient (cm_production_gradient_cplus -> build_lfix_base_cache_C! -> build_winner_ref!,
# lfix_factorized_workspace.jl) already solves this exact problem: it stores only the COMPACT
# logCC0 (D x Ddest) and mulU (W x D) tables (mirroring exactly what constCons_matrix/price
# formulas need), and computes any origin's score/price ON THE FLY wherever needed
# (`logCC0[o,d]+mulU[w,o]`, `pTσ_from_score(score,σ)=exp((1-σ)*score)`) instead of materializing a
# dense W x D x Ddest tensor. Confirmed mathematically identical to the REDUCED formulation's own
# price/pTσ (price = constCons_od*U^μ, so log(price) = log(constCons_od)+μ*log(U) = logCC0+mulU
# exactly; pTσ_from_score(log(price),σ) = price^(1-σ), matching price_and_pTsigma_cell's own pTσ
# formula given the model's Uσ=U^(1-σ) convention) -- verified by cross-reading both formulas
# line-by-line, not assumed. This section REUSES build_winner_ref!/pTσ_from_score verbatim (no new
# kernel, no re-derivation) via a persistent, lazily-(re)built workspace, exactly the same
# ensure_*!-workspace idiom used throughout this codebase.
"Persistent, lazily-(re)built workspace backing every profiled/reduced family's shared gradient engine -- built ONCE per (D,Ddest,W), refilled (never reallocated) on every subsequent call, mirroring cm_production_gradient_cplus's own LFixFactorizedWorkspace exactly. Module-level Ref (same discipline as NO_DENSE_G_COUNTERS/ZC_EZ_BACKEND_DEFAULT): only one family's outer search is ever active per process."
const SHARED_PROFILED_LFIX_WS_REF = Ref{Union{Nothing,LFixFactorizedWorkspace}}(nothing)

"Nullable-ref-safe wrapper around `ensure_lfix_factorized_workspace!` (which requires an already-built, non-nullable `Base.RefValue{LFixFactorizedWorkspace}`) -- builds fresh on first use or a genuine (D,Ddest,W) change, otherwise returns the existing persistent workspace unchanged."
function ensure_shared_profiled_lfix_ws!(D::Int, Ddest::Int, W::Int)
    ws = SHARED_PROFILED_LFIX_WS_REF[]
    if ws === nothing || ws.D != D || ws.Ddest != Ddest || ws.W != W
        ws = build_lfix_factorized_workspace(D, Ddest, W)
        SHARED_PROFILED_LFIX_WS_REF[] = ws
    end
    return ws
end

"""
    build_price_winner_base_cache(ctx, x_free0, θ_full, cf) -> NamedTuple

Performance closeout task (2026-08-02): REPLACES the former dense-tensor implementation (which
allocated fresh `W x D x Ddest` price0/pTσ0 arrays on every call -- the confirmed root cause of the
~6.5-9s/call gradient-engine cost measured during the Section 13 outer-search investigation) with a
direct call to FULL's own already-optimized, already-validated `build_winner_ref!`
(lfix_factorized_workspace.jl) against a persistent workspace -- same top3_scan winner-finding
algorithm, same score formula, only the STORAGE differs (compact logCC0/mulU instead of a dense
tensor). `x_free0` is the free-parameter vector `build_winner_ref!` itself needs (it does its own
`CS.reconstruct_full` internally); callers already have this as `ev.decoded.xf`. Retains the
former's live cross-check (`winner0 == cf.winner`) as a correctness safeguard.
"""
function build_price_winner_base_cache(ctx, x_free0::AbstractVector, θ_full::AbstractVector, cf)
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    σ = θ_full[2]

    ws = ensure_shared_profiled_lfix_ws!(D, Ddest, W)
    ref = build_winner_ref!(ws, x_free0, ctx)

    ref.winner == cf.winner || error("build_price_winner_base_cache: rebuilt winner (via build_winner_ref!) disagrees with cf.winner -- constCons_matrix/build_compressed_factual formulas have diverged")

    # third_pTσ0 -- the only quantity build_winner_ref! doesn't already provide directly (it keeps
    # third's SCORE in ref.st3 but the downstream contrib formula needs third's pTσ specifically at
    # the France/cf row) -- one O(W*Ddest) pTσ_from_score call, cheap (no tensor materialization).
    third_pTσ0 = Matrix{Float64}(undef, W, Ddest)
    @inbounds for d in 1:Ddest, ω in 1:W
        t = ref.third[ω, d]
        third_pTσ0[ω, d] = t == 0 ? Inf : pTσ_from_score(ref.st3[ω, d], σ)
    end

    return (logCC0 = ref.logCC0, mulU = ref.mulU, winner0 = ref.winner, winner_price0 = ref.sw,
        runnerup0 = ref.runnerup, runnerup_price0 = ref.sr, third0 = ref.third, third_price0 = ref.st3,
        third_pTσ0 = third_pTσ0)
end

function build_profiled_lfix_cache(w_profiled::AbstractVector{Float64}, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, ev)
    st = ev.st; cf = st.cf; layout = st.layout; θ_full = ev.theta_full; obj = ev.obj
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    μ = θ_full[1]; σ = θ_full[2]
    SW = cf.SW

    base = build_price_winner_base_cache(ctx, ev.decoded.xf, θ_full, cf)
    logCC0 = base.logCC0; mulU = base.mulU
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
        # winner_price0[ω,d] IS the winner's own SCORE (== logCC0[wo,d]+mulU[ω,wo] by construction,
        # from build_winner_ref!) -- pTσ_from_score converts it to pTσ directly, on the fly, no
        # dense tensor lookup (mirrors build_lfix_base_cache_C!'s own contrib0 loop exactly,
        # lfix_factorized_workspace.jl:173-179).
        contrib0[ω, d] = (κ[wo, d] - Cbar_eff[d]) * pTσ_from_score(winner_price0[ω, d], σ)
    end

    const_part = const_cf - pmmterm
    q0 = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        acc = const_part + sum(@view contrib0[w, :]) + (has_france ? cf_raw_κcf[w] : 0.0)
        t0 = SW[w] * acc
        q0[w] = -ev.result.zeta - t0
    end

    return ProfiledLFixCache(D, Ddest, W, μ, σ, SW, logCC0, mulU, winner0, winner_price0, runnerup0,
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

O(1)-per-draw winner update (top-3-cache, at most 2 changed origins in one destination -- direct+
pivot never touches more), producing `(kappa[wo,d]-Cbar_eff[d])*pTsigma[wo]` per draw.

Performance closeout task (2026-08-02): mirrors FULL's own `dest_contrib_incremental_top3_C`
(lfix_factorized.jl) exactly -- same on-the-fly score computation (`logCC_new[o,d]+cache.mulU[ω,o]`,
`constCons_matrix`/`pTσ_from_score` both reused unmodified), same top-3-cache-then-rescan structure,
same exact-tie convention (lowest origin index wins). No dense price0/pTσ0 tensor read anywhere --
only the FINAL contribution formula differs from FULL's (REDUCED's own kappa/Cbar_eff dual-based
formula, not FULL's lambda-star one).
"""
function dest_contrib_reduced_o1(cache::ProfiledLFixCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    length(changed_origins) <= 2 || error("dest_contrib_reduced_o1: >2 changed origins in one destination is unreachable for a single profiled coordinate (direct+pivot only)")
    D = cache.D; W = cache.W; σ = cache.σ
    κd = @view cache.κ[:, d]; Cd = cache.Cbar_eff[d]
    Cd_set = changed_origins
    _, logCC_new, _ = constCons_matrix(θ_full, ctx)
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_s = Inf
        if !(r1 in Cd_set)
            best_o = r1; best_s = cache.winner_price0[ω, d]
        elseif !(r2 in Cd_set)
            best_o = r2; best_s = cache.runnerup_price0[ω, d]
        elseif r3 != 0 && !(r3 in Cd_set)
            best_o = r3; best_s = cache.third_price0[ω, d]
        end
        bo = best_o; bs = best_s
        for o in Cd_set
            v = logCC_new[o, d] + cache.mulU[ω, o]
            if v < bs || (v == bs && o < bo)
                bs = v; bo = o
            end
        end
        if bo == 0
            # extremely defensive: all of top-3 were changed (D<=3 & |Cd_set|>=3) -- unreachable for
            # |Cd_set|<=2 with D>=3, same defensive fallback dest_contrib_incremental_top3_C has.
            bo = 1; bs = logCC_new[1, d] + cache.mulU[ω, 1]
            for o in 2:D
                v = logCC_new[o, d] + cache.mulU[ω, o]
                v < bs && (bs = v; bo = o)
            end
        end
        contrib[ω] = (κd[bo] - Cd) * pTσ_from_score(bs, σ)
    end
    return contrib
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
        Tslot_bi += cache.SW[ω] * m_weights[ω] * pTσ_from_score(cache.winner_price0[ω, cache.bi_slot], cache.σ)
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
    length(changed_origins) <= 2 || error("profiled_count_winner_flips: >2 changed origins unreachable for a single profiled coordinate")
    D = cache.D; W = cache.W
    Cd = changed_origins
    # Performance closeout task (2026-08-02): mirrors FULL's own count_winner_flips_C
    # (lfix_factorized.jl) exactly -- same on-the-fly score computation, no dense price0 tensor.
    _, logCC_new, _ = constCons_matrix(θ_full, ctx)
    flips = 0
    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_s = Inf
        if !(r1 in Cd)
            best_o = r1; best_s = cache.winner_price0[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_s = cache.runnerup_price0[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_s = cache.third_price0[ω, d]
        end
        bo = best_o; bs = best_s
        for o in Cd
            v = logCC_new[o, d] + cache.mulU[ω, o]
            if v < bs || (v == bs && o < bo)
                bs = v; bo = o
            end
        end
        if bo == 0
            bo = 1; bs = logCC_new[1, d] + cache.mulU[ω, 1]
            for o in 2:D
                v = logCC_new[o, d] + cache.mulU[ω, o]
                v < bs && (bs = v; bo = o)
            end
        end
        flips += (bo != r1)
    end
    return flips
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
        pe::PivotGravityElimOnRetained, ev; threaded::Bool)
    cache = build_profiled_lfix_cache(w_profiled, ctx, spec, pe, ev)
    return profiled_composite_gradient_from_cache(cache, ctx, spec, pe, w_profiled, ev; threaded = threaded)
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
        pe::PivotGravityElimOnRetained, w_profiled::AbstractVector{Float64}, ev;
        threaded::Bool)
    n_total = outer_dim_profiled(pe)
    g = zeros(n_total)
    g[1] = profiled_gp_component_analytic(cache, w_profiled, ev, ctx)

    h_used = zeros(n_total); switch_mass = zeros(n_total)

    # task §6 (profiled-outer-ab-readiness-2026-08-04): port of FULL's own
    # composite_gradient_fast.jl::do_coord!/Threads.@threads structure. Each coordinate's work
    # writes ONLY to its own index `coord_idx` of `g`/`h_used`/`switch_mass` (pre-allocated,
    # thread-safe by construction -- no shared mutable state). `cache`/`w_profiled`/`ctx`/`spec`/
    # `pe` are read-only for the duration of the loop (`profiled_lfix_incremental_at`/
    # `profiled_select_bandwidth` each build their own local `w = copy(w0)` and fresh allocations
    # internally -- confirmed by direct read, not assumed). No cross-coordinate cache/dict is
    # touched here (unlike FULL's h_mode=:cached path, which needs its own lock) -- REDUCED's
    # bandwidth selection has no analogous shared cache yet (task §7).
    function do_coord!(coord_idx::Int)
        h, m, _selmeta = profiled_select_bandwidth(cache, ctx, spec, pe, w_profiled, coord_idx)
        h_used[coord_idx] = h; switch_mass[coord_idx] = m
        Lp = profiled_lfix_incremental_at(cache, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] + h)
        Lm = profiled_lfix_incremental_at(cache, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] - h)
        g[coord_idx] = (Lp - Lm) / (2h)
        return nothing
    end

    if threaded
        # Same mutual-exclusion invariant FULL's own threaded coordinate pool enforces
        # (parallelism_guards.jl, injected into CS by context.jl -- already loaded by every
        # REDUCED include list, no new include needed): errors loudly if an inner KNITRO solve is
        # somehow still active when this pool launches, rather than silently racing.
        Main.CS.guard_enter_coord_pool!()
        try
            Threads.@threads for coord_idx in 2:n_total
                do_coord!(coord_idx)
            end
        finally
            Main.CS.guard_exit_coord_pool!()
        end
    else
        @inbounds for coord_idx in 2:n_total
            do_coord!(coord_idx)
        end
    end
    return g, (cache = cache, w0 = collect(Float64, w_profiled), h_used = h_used, switch_mass = switch_mass,
        threaded = threaded)
end
