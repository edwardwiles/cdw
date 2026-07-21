# ============================================================================
# Reusable immutable-context helpers (task §5). Factors the ctx/pe/rsc construction
# that run_profile_checkpointed / run_polish_checkpointed each redo on EVERY call
# (the real cost: ~65-83s of draw generation + screen setup per the production
# consolidation handoff §10/staged-continuation finding) into a standalone builder
# that a caller can invoke ONCE and thread through multiple stages.
#
# Immutable-context / stage-specific split (task §5):
#   - IMMUTABLE, reusable across stages: draws, transformed productivity arrays,
#     gravity maps/parameter ordering, moment layout, pivot-elimination map (pe),
#     ranged-screen context (rsc) -- confirmed δ-INDEPENDENT by inspection
#     (build_pivot_elimination/build_ranged_screen_context reference neither ctx.δ
#     nor ctx.obj.δ anywhere in gravity_elimination.jl/fast_range_screen.jl), QMC/draw
#     metadata, destination constants.
#   - STAGE-SPECIFIC: δ itself, the outer starting point, the outer KNITRO context,
#     the best-feasible incumbent, checkpoint path/budget.
#
# One real subtlety this module exists to handle correctly: δ is NOT stored only on
# `ctx.δ` -- the actual inner CC divergence-budget check and the exact-point cache key
# (FullAEvalKey, oracle.jl) both read `ctx.obj.δ`, a field on the mutable
# PsiObjectiveBundleImplicit (`ctx.obj`), NOT the immutable ctx NamedTuple's own `.δ`
# field. A naive `merge(ctx, (δ=new_δ,))` would silently leave `ctx.obj.δ` stale,
# corrupting both the feasibility check every downstream screened_eval/cb_F! call makes
# AND the exact-point cache's own key. `set_context_delta!` updates both consistently.
# ============================================================================

"""
    build_fullA_context(; W, δ, find_smallest, draw_design, draw_seed, inner_loop_opt=nothing)
        -> (ctx=ctx, pe=pe, rsc=rsc)

One-time real-data context build: the same `d20_real_setup_design` +
`build_pivot_elimination` + `build_ranged_screen_context` sequence
run_profile_checkpointed/run_polish_checkpointed each already do internally, factored out
so a caller can build it ONCE and reuse it across multiple `run_*_checkpointed` calls via
the `reuse=` keyword.
"""
function build_fullA_context(; W::Int, δ::Float64, find_smallest::Bool,
        draw_design::Symbol, draw_seed::Int, inner_loop_opt::Union{Nothing,AbstractString} = nothing)
    ctx = inner_loop_opt === nothing ?
        d20_real_setup_design(W = W, δ = δ, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed) :
        d20_real_setup_design(W = W, δ = δ, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed, inner_loop_opt = inner_loop_opt)
    pe = build_pivot_elimination(ctx)
    rsc = build_ranged_screen_context(ctx)
    return (ctx = ctx, pe = pe, rsc = rsc)
end

"""
    set_context_delta!(ctx, new_delta::Float64) -> ctx′

Overrides the divergence budget on an existing context for reuse at a new δ stage.
Mutates `ctx.obj.δ` in place (the field the inner solve / exact-cache key actually read
-- see module docstring) AND returns a new NamedTuple with `.δ` merged in (the field the
driver's own `KN_set_con_upbnd`/`feasible = Δ <= ctx.δ + 1e-6` checks read). `pe`/`rsc`
are untouched -- confirmed δ-independent, see module docstring -- and remain valid as-is.

This is a cheap O(1) operation (a NamedTuple `merge` shares all other fields' underlying
arrays by reference, no data is copied) -- NOT a context rebuild.
"""
function set_context_delta!(ctx, new_delta::Float64)
    ctx.obj.δ = new_delta
    return merge(ctx, (δ = new_delta,))
end

"""
    reuse_matches(reuse::NamedTuple; W, find_smallest, draw_design, draw_seed) -> Bool

Guard used by run_*_checkpointed's `reuse=` path: a reused context must have been built
under the SAME W/find_smallest/draw_design/draw_seed the caller is now requesting (δ is
exempt -- that's exactly the field reuse is meant to override). Prevents silently reusing
a context built for a different problem instance.
"""
function reuse_matches(reuse::NamedTuple; W::Int, find_smallest::Bool, draw_design::Symbol, draw_seed::Int)
    ctx = reuse.ctx
    return ctx.W == W && ctx.find_smallest == find_smallest &&
           ctx.draw_design == draw_design && ctx.draw_seed == draw_seed
end
