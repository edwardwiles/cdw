# ============================================================================
# Part B restoration (2026-07-23 release): wires the existing exact
# pairwise/hard-winner/witness infeasibility screens (infeasibility_screen.jl)
# into the CM/origin-ZC production evaluation path.
#
# docs/SCREEN_STACK_AUDIT_2026-07-23.md found `cm_production_value_verified` /
# `cm_originzc_production_value_verified` (cm_production_bundle.jl) call
# `archC_base_state`/`archC_verified_state` -> `inner_loop_internal_archgeneric`
# DIRECTLY, bypassing every screen the base D=20 driver
# (c10_d20_production_driver.jl::screened_eval) already runs by default. This
# was a wiring gap from when the CM bundle was built, not a regression from
# previously-active CM screening (screens were never reachable from this path).
#
# ADDITIVE ONLY: does not modify archC_base_state/archC_verified_state/
# cm_production_value*/cm_originzc_production_value_verified
# (cm_production_bundle.jl / cm_originzc_moments.jl) -- new wrapper functions
# only. `ctx_cm`/`ctx_oz` already carry `.pairwise`/`.witness`, precomputed
# ONCE at context-construction time by context_real_d20.jl::d20_real_setup
# (build_screen=true default) and inherited unchanged through
# `merge(ctx, (obj=obj_cm,))` in build_cm_production_context -- this file only
# adds the missing QUERY of those already-built certificates before the inner
# solve, reusing pairwise_certificate/screen_hard_winners/query_witness
# exactly as screened_eval already does, so screen-on and screen-off return
# byte-identical verified values on feasible points (same certificate
# functions, same ctx fields, same math -- not a re-derivation).
#
# Reduced-destination requirement (Part B step 8): the screens iterate over
# `ctx_cm.D`/`ctx_cm.U`, i.e. the base context's ACTIVE destination set only
# -- a missing/impossible winner for an already-omitted destination (e.g. a
# dropped ROW column, Part A) is not part of that loop and cannot trigger a
# rejection it shouldn't.
# ============================================================================

mutable struct CMScreenCounters
    pairwise::Int
    witness::Int
    winner::Int
    passed::Int
end
CMScreenCounters() = CMScreenCounters(0, 0, 0, 0)
as_namedtuple(sc::CMScreenCounters) = (pairwise = sc.pairwise, witness = sc.witness, winner = sc.winner, passed = sc.passed)

"""
    cm_screen_precheck!(x_free0, ctx_cm; counters=nothing, use_witness=false) -> Nothing

Throws `CMExpectedSolveFailure` (the SAME typed exception every existing CM/ZC caller already
catches via `e isa CMExpectedSolveFailure`) the instant a draw-free EXACT certificate proves the
outer point infeasible, before any KNITRO inner solve is attempted. No-op (returns `nothing`,
falls through to the unscreened inner solve unchanged) if the point passes every screen, or if
`ctx_cm.pairwise === nothing` (this ctx was built with `build_screen=false` -- degrades to the
pre-restoration behavior rather than silently changing results for a ctx that opted out).

Per task Part B step 9: this is an exact mathematical certificate (never a heuristic/numerical
one) -- it may only ever be caused by the current outer point being genuinely infeasible on the
fixed draw support, never by a KNITRO timeout, a -300 status, or an approximate inequality.
"""
function cm_screen_precheck!(x_free0::AbstractVector, ctx_cm;
                              counters::Union{Nothing,CMScreenCounters} = nothing,
                              use_witness::Bool = false)
    ctx_cm.pairwise === nothing && return nothing
    θ_full = CS.reconstruct_full(x_free0, ctx_cm.m)
    Pmat = target_shares(ctx_cm)
    a = compute_a_od(θ_full, ctx_cm)

    pres = pairwise_certificate(a, ctx_cm.pairwise, Pmat)
    if pres.infeasible
        counters !== nothing && (counters.pairwise += 1)
        throw(CMExpectedSolveFailure("cm_screen_precheck!: pairwise-certified infeasible at (o=$(pres.worst_o), d=$(pres.worst_d))"))
    end

    if use_witness && ctx_cm.witness !== nothing
        B = hard_score_B(ctx_cm)
        for d in 1:ctx_cm.D, o in 1:ctx_cm.D
            Pmat[o, d] > 0 || continue
            exists, _, _, _ = query_witness(o, d, a, B, ctx_cm.witness)
            if !exists
                counters !== nothing && (counters.witness += 1)
                throw(CMExpectedSolveFailure("cm_screen_precheck!: witness-certified infeasible at (o=$o, d=$d)"))
            end
        end
    end

    order = order_destinations(pres, ctx_cm.D)
    wres = screen_hard_winners(θ_full, ctx_cm, Pmat; order = order)
    if !wres.feasible
        counters !== nothing && (counters.winner += 1)
        throw(CMExpectedSolveFailure("cm_screen_precheck!: zero-winner certified infeasible at (o=$(wres.failing_o), d=$(wres.failing_d))"))
    end

    counters !== nothing && (counters.passed += 1)
    return nothing
end

"""
    archC_base_state_screened(x_free0, ctx_cm, cctx; counters=nothing, use_witness=false) -> BaseDualState

Screened drop-in replacement for `archC_base_state` (cm_production_bundle.jl): runs
`cm_screen_precheck!` first, then falls through to the existing, unmodified `archC_base_state`.
An infeasible-by-screen point raises the same `CMExpectedSolveFailure` a failed inner solve
would have, so no caller needs to change its exception handling.
"""
function archC_base_state_screened(x_free0::AbstractVector, ctx_cm, cctx;
                                    counters::Union{Nothing,CMScreenCounters} = nothing,
                                    use_witness::Bool = false)
    cm_screen_precheck!(x_free0, ctx_cm; counters = counters, use_witness = use_witness)
    return archC_base_state(x_free0, ctx_cm, cctx)
end

"""
    archC_verified_state_screened(x_free0, ctx_cm, cctx; counters=nothing, use_witness=false) -> (base, verify)

Screened drop-in replacement for `archC_verified_state`, same contract as
`archC_base_state_screened`.
"""
function archC_verified_state_screened(x_free0::AbstractVector, ctx_cm, cctx;
                                        counters::Union{Nothing,CMScreenCounters} = nothing,
                                        use_witness::Bool = false)
    cm_screen_precheck!(x_free0, ctx_cm; counters = counters, use_witness = use_witness)
    return archC_verified_state(x_free0, ctx_cm, cctx)
end

"""
    cm_production_value_verified_screened(x_free0, pcx; counters=nothing, use_witness=false) -> (K, base, verify)

Screened drop-in replacement for `cm_production_value_verified` -- the function
`run_cm_upper_checkpointed` (cm_checkpoint.jl) now calls (Part B restoration wiring, see that
file's call sites). Same return contract.
"""
function cm_production_value_verified_screened(x_free0::AbstractVector, pcx;
                                                 counters::Union{Nothing,CMScreenCounters} = nothing,
                                                 use_witness::Bool = false)
    base, verify = archC_verified_state_screened(x_free0, pcx.ctx_cm, pcx.cctx; counters = counters, use_witness = use_witness)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end

"""
    cm_meanzc_production_value_verified_screened(x_free0, νvec, pcx; counters=nothing, use_witness=false) -> (K, base, verify)

Screened drop-in replacement for `cm_meanzc_production_value_verified` (cm_meanzc_production.jl,
CM + mean/ZC restriction family) -- same `cm_screen_precheck!` pre-check as the flexible-CM
wrapper above, then falls through to the existing `archC_meanzc_verified_state` unchanged.
"""
function cm_meanzc_production_value_verified_screened(x_free0::AbstractVector, νvec::AbstractVector{Float64}, pcx;
                                                        counters::Union{Nothing,CMScreenCounters} = nothing,
                                                        use_witness::Bool = false)
    cm_screen_precheck!(x_free0, pcx.ctx_cm; counters = counters, use_witness = use_witness)
    base, verify = archC_meanzc_verified_state(x_free0, νvec, pcx.ctx_cm, pcx.cctx)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end

"""
    cm_originzc_production_value_verified_screened(x_free0, νfull, pcx; counters=nothing, use_witness=false) -> (K, base, verify)

Screened drop-in replacement for `cm_originzc_production_value_verified`
(cm_originzc_production.jl, origin-specific-ZC restriction family) -- same `cm_screen_precheck!`
pre-check, then falls through to the existing `archOZ_verified_state` unchanged.
"""
function cm_originzc_production_value_verified_screened(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx;
                                                          counters::Union{Nothing,CMScreenCounters} = nothing,
                                                          use_witness::Bool = false)
    cm_screen_precheck!(x_free0, pcx.ctx_cm; counters = counters, use_witness = use_witness)
    base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end
