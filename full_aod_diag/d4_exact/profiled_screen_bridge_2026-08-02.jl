# ============================================================================
# Phase 12 (integration/phase12-13-runner-checkpoints-2026-08-02), item 1: reuse the
# existing scientific screen logic (cm_screen_bridge.jl / infeasibility_screen.jl) for the
# profiled/reduced outer coordinate `w_profiled`, after decoding to the equivalent FULL
# state -- not a re-derivation of any certificate math, and not a new indexing scheme.
#
# WHY this is safe, not just convenient: `decode_outer_profiled` (outer_coordinate_layout_
# profiled_2026-07-31.jl) already produces `xf` in EXACTLY the shape `decode_outer_unified`
# produces for the dense/full formulation -- that file's own docstring says so explicitly
# ("downstream screened-evaluation ... code that only reads xf needs no changes at all").
# `cm_screen_precheck!` (cm_screen_bridge.jl) itself never touches the REDUCED economic
# moment layout at all -- it reconstructs `θ_full` from `x_free0` via `CS.reconstruct_full`,
# then does every certificate check (`pairwise_certificate`, `screen_hard_winners`,
# `query_witness`) against FULL (o,d) destination indices (`ctx_cm.D`/`target_shares(ctx_cm)`).
# So the correct, minimal wiring is: decode `w_profiled` -> `xf` (existing, gated function),
# then call the existing full-space screen unchanged on that `xf`. `ProfiledEconomicMomentLayout`'s
# `retained_full_factual_j`/`full_factual_to_reduced`/`reduced_to_full_factual` maps are a
# separate concern (they translate REDUCED ECONOMIC MOMENT indices <-> full-space moment j,
# used by the shared outer-gradient engine's own kappa/Cbar accumulation) -- irrelevant to this
# wrapper specifically BECAUSE the screen operates on full θ/full (o,d) cells, never on the
# reduced economic dual vector at all. This file's gate (test_profiled_screen_bridge_reduced_
# layout_2026-08-02.jl) proves this claim directly (A/B: same underlying point, screened via
# the reduced-decode path vs. fed directly in full space, byte-identical certificate outcome)
# rather than merely asserting it.
#
# ADDITIVE ONLY -- does not modify cm_screen_bridge.jl / infeasibility_screen.jl /
# outer_coordinate_layout_profiled_2026-07-31.jl.
# ============================================================================

isdefined(Main, :cm_screen_precheck!) ||
    error("profiled_screen_bridge_2026-08-02.jl requires cm_screen_bridge.jl to be included first.")
isdefined(Main, :decode_outer_profiled) ||
    error("profiled_screen_bridge_2026-08-02.jl requires outer_coordinate_layout_profiled_2026-07-31.jl to be included first.")

"""
    profiled_cm_screen_precheck!(w_profiled, ctx, pe; counters=nothing, use_witness=false) -> Nothing

Reduced/profiled-coordinate sibling of `cm_screen_precheck!`. Decodes `w_profiled` to the full
free-parameter vector `decoded.xf` via the EXISTING, already-gated `decode_outer_profiled`, then
calls the EXISTING `cm_screen_precheck!` on that `xf` UNCHANGED -- no new certificate math, no new
indexing scheme. Throws the SAME `CMExpectedSolveFailure` `cm_screen_precheck!` would throw for
the equivalent full-space point; no-ops (falls through to the caller's own inner solve) under the
same conditions `cm_screen_precheck!` does (`ctx.pairwise === nothing`, i.e. this ctx was built
with `build_screen=false`).
"""
function profiled_cm_screen_precheck!(w_profiled::AbstractVector{Float64}, ctx, pe::PivotGravityElimOnRetained;
        counters::Union{Nothing,CMScreenCounters} = nothing, use_witness::Bool = false)
    decoded = decode_outer_profiled(collect(Float64, w_profiled), ctx, pe)
    return cm_screen_precheck!(decoded.xf, ctx; counters = counters, use_witness = use_witness)
end
