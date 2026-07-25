# ============================================================================
# Allocation/Hessian port task, section 3.3: hard_score_B(ctx) = -log.(ctx.U) is a pure function
# of ctx.U, which never changes for the whole outer-solve process (draw-free, per its own
# docstring in infeasibility_screen.jl) -- yet was recomputed from scratch (fresh W x D allocation
# + a full elementwise log) on every witness-screen call. Cache it once per ctx.
# ============================================================================

"""
    attach_hard_score_b_cache(ctx) -> ctx

Returns `ctx` merged with a `hard_score_B_cache::Matrix{Float64}` field (`-log.(ctx.U)`, built
once). `hard_score_B` (infeasibility_screen.jl) serves this directly when present. Call once per
outer-solve process, alongside `attach_compressed_factual_workspace`/
`attach_canonical_price_precompute_workspace`.
"""
function attach_hard_score_b_cache(ctx)
    hasproperty(ctx, :hard_score_B_cache) && return ctx
    return merge(ctx, (hard_score_B_cache = -log.(ctx.U),))
end
