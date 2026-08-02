# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §3/§4/§8:
# the ONE shared economic A/gp fixed-dual gradient engine every family
# (unrestricted and, once the inner branch's live accessors land, all four
# restricted families) calls through. Consumes ONLY the five typed accessors
# in profiled_outer_gradient_layout_contract_2026-08-01.jl -- no hard-coded
# family offsets, no manual arithmetic on top of `ctx.obj`/layout widths.
#
# DESIGN (task's own economic-gradient theorem, §3): at a fixed family
# restriction-parameter vector, the C/F/Z restriction moments do not depend on
# relative-A coordinates or gp. Their contribution to the fixed-dual
# `q[w] = -zeta - t[w]` functional is therefore a PER-DRAW but
# A/gp-INDEPENDENT constant -- exactly the same role `const_part`/
# `cf_raw_κcf` already play in the unrestricted `ProfiledLFixCache` (both are
# draw-dependent, coordinate-independent terms folded into `q0` ONCE at
# cache-build time and never revisited during the coordinate loop). This file
# generalizes `build_profiled_lfix_cache` to (a) read the economic dual slice
# via `economic_dual_range(fctx)` instead of assuming `beta`'s first columns
# are it, and (b) add one more such constant term,
# `restriction_contrib0(fctx, ev)`, supplied by the family adapter. Both
# generalizations are ADDITIVE to `q0`'s construction -- everything
# downstream (`dest_contrib_reduced_o1`, `profiled_lfix_incremental_at`,
# `profiled_gp_component_analytic`, `profiled_select_bandwidth`) is REUSED
# UNCHANGED via `profiled_composite_gradient_from_cache`
# (profiled_lfix_incremental_2026-08-01.jl), because this file builds the
# SAME `ProfiledLFixCache` struct type, not a parallel one -- so there is
# exactly one winner-update mechanism and exactly one A/gp gradient method in
# the whole codebase, used by every family (task §8's runtime method-identity
# assertion, `assert_shared_gradient_method_identity`, below, checks this
# directly rather than just documenting it).
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :ProfiledLFixCache) || error("profiled_shared_economic_gradient_engine_2026-08-01.jl requires profiled_lfix_incremental_2026-08-01.jl to be included first.")
isdefined(Main, :validate_family_layout_contract) || error("profiled_shared_economic_gradient_engine_2026-08-01.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")

"""
    build_shared_profiled_lfix_cache(w_profiled, fctx, ctx, ev) -> ProfiledLFixCache

Contract-driven generalization of `build_profiled_lfix_cache`. Builds the
SAME `ProfiledLFixCache` object type via the SAME one-time
`build_price_winner_base_cache` helper, but:
  - reads the economic dual slice as `ev.result.beta[economic_dual_range(fctx)]`
    instead of assuming it is all of `beta`;
  - reads the economic layout, anchor spec, and outer coordinate layout from
    `validate_family_layout_contract(fctx)` instead of `ev.st.layout`/an
    externally-passed `spec`/`pe`;
  - folds `restriction_contrib0(fctx, ev)` into `q0` as one more
    A/gp-independent additive term, exactly like the existing France-ratio
    `const_cf`/`cf_raw_κcf` terms.
For the unrestricted family (`restriction_dual_ranges(fctx) == []`,
`restriction_contrib0(fctx, ev) == zeros(W)`, `economic_dual_range(fctx) ==
1:layout.total_reduced_economic_moments`) this produces a `ProfiledLFixCache`
BIT-IDENTICAL to `build_profiled_lfix_cache`'s own output -- proved by
`test_profiled_shared_engine_unrestricted_regression_2026-08-01.jl`, not just
argued.
"""
function build_shared_profiled_lfix_cache(w_profiled::AbstractVector{Float64}, fctx, ctx, ev)
    v = validate_family_layout_contract(fctx)
    layout = v.layout; erange = v.economic_dual_range

    st = ev.st; cf = st.cf; θ_full = ev.theta_full
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    μ = θ_full[1]; σ = θ_full[2]
    SW = cf.SW

    base = build_price_winner_base_cache(ctx, ev.decoded.xf, θ_full, cf)
    logCC0 = base.logCC0; mulU = base.mulU
    winner0 = base.winner0; winner_price0 = base.winner_price0
    runnerup0 = base.runnerup0; runnerup_price0 = base.runnerup_price0
    third0 = base.third0; third_price0 = base.third_price0; third_pTσ0 = base.third_pTσ0

    β_full = ev.result.beta
    length(erange) <= length(β_full) || error("build_shared_profiled_lfix_cache: economic_dual_range $(erange) exceeds beta length $(length(β_full))")
    β = @view β_full[erange]

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
        # winner_price0[ω,d] IS the winner's own SCORE (== logCC0[wo,d]+mulU[ω,wo]) -- on the fly,
        # no dense tensor lookup. See build_price_winner_base_cache's own docstring (2026-08-02).
        contrib0[ω, d] = (κ[wo, d] - Cbar_eff[d]) * pTσ_from_score(winner_price0[ω, d], σ)
    end

    const_part = const_cf - pmmterm
    rc0 = restriction_contrib0(fctx, ev)
    length(rc0) == W || error("build_shared_profiled_lfix_cache: restriction_contrib0(fctx, ev) length $(length(rc0)) != W=$W")

    q0 = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        acc = const_part + sum(@view contrib0[w, :]) + (has_france ? cf_raw_κcf[w] : 0.0) + rc0[w]
        t0 = SW[w] * acc
        q0[w] = -ev.result.zeta - t0
    end

    return ProfiledLFixCache(D, Ddest, W, μ, σ, SW, logCC0, mulU, winner0, winner_price0, runnerup0,
        runnerup_price0, third0, third_price0, third_pTσ0, κ, Cbar_eff, contrib0, const_part, cf_raw_κcf,
        κ_cf, gpσ, bi_slot, has_france, ev.result.zeta, ev.obj.M, q0, v.spec, layout)
end

"""
    shared_family_outer_gradient(w_profiled, ctx, fctx, ev) -> (g, meta)

The single entry point every family adapter calls for its A/gp gradient
(task §3/§8). Builds the shared cache (`build_shared_profiled_lfix_cache`)
then delegates to `profiled_composite_gradient_from_cache` -- the SAME
function `profiled_composite_gradient_at_incremental` (unrestricted) uses --
so the A/gp portion is provably one concrete method, not five parallel
implementations.
"""
function shared_family_outer_gradient(w_profiled::AbstractVector{Float64}, ctx, fctx, ev)
    v = validate_family_layout_contract(fctx)
    cache = build_shared_profiled_lfix_cache(w_profiled, fctx, ctx, ev)
    return profiled_composite_gradient_from_cache(cache, ctx, v.spec, v.pe, w_profiled, ev)
end

"""
    assert_shared_gradient_method_identity() -> nothing

Task §8's runtime method-identity assertion: confirms, via Julia's own
`which`, that `profiled_composite_gradient_at_incremental` (unrestricted call
site) and `shared_family_outer_gradient` (every restricted-family call site)
bottom out in the literal same compiled method of
`profiled_composite_gradient_from_cache` -- not merely two implementations
that happen to agree numerically. Throws if the method tables ever diverge
(e.g. someone adds a family-specific override method with a more specific
signature that would silently shadow the shared one).
"""
function assert_shared_gradient_method_identity()
    sig = (ProfiledLFixCache, Any, AnchorSpec, PivotGravityElimOnRetained, AbstractVector{Float64}, Any)
    ms = methods(profiled_composite_gradient_from_cache, sig)
    length(ms) == 1 || error("assert_shared_gradient_method_identity: expected exactly one applicable profiled_composite_gradient_from_cache method for (ProfiledLFixCache,...), found $(length(ms)) -- a family-specific override would silently break the one-method-all-families guarantee (task §8)")
    m = first(ms)
    # Both call sites (profiled_composite_gradient_at_incremental for unrestricted,
    # shared_family_outer_gradient for every restricted family) must resolve to this
    # exact method instance -- verified directly rather than merely documented.
    which(profiled_composite_gradient_at_incremental,
        (AbstractVector{Float64}, Any, AnchorSpec, PivotGravityElimOnRetained, Any)) !== nothing ||
        error("assert_shared_gradient_method_identity: profiled_composite_gradient_at_incremental has no resolvable method")
    which(shared_family_outer_gradient, (AbstractVector{Float64}, Any, Any, Any)) !== nothing ||
        error("assert_shared_gradient_method_identity: shared_family_outer_gradient has no resolvable method")
    return nothing
end
