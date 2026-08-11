# ============================================================================
# Allocation/Hessian port task, section 3.3: canonical_price_precompute (winner_certificate.jl)
# allocated fresh W x D `mulU`/`UPow`/`UσPow` matrices on EVERY call (audit's own dynamic
# Profile.Allocs by-site trace: winner_certificate.jl:123-125, ~12.8 MB/event EACH, ~150 MB/call
# combined at real D=20/W=80,000). `mulU`'s own `log.(U)` sub-expression is additionally pure
# recomputation waste: U never changes for the whole campaign, only the scalar mu multiplying it
# varies per call -- so `log.(U)` itself needs computing exactly ONCE ever, not per call.
#
# Reentrancy checked (required before sharing one workspace across multiple call sites): all four
# callers of canonical_price_precompute (screen_hard_winners_ranged, screen_hard_winners,
# build_compressed_factual/build_compressed_factual!) extract mulU/UPow/UσPow into local bindings,
# consume them ONLY within their own loop, and return before the function ends -- none retain them
# past their own call, and none of these callers are themselves reentrant/nested with each other
# within a single screened_eval invocation (they run sequentially: a screen either rejects and
# returns, or completes and the NEXT screen/call runs). So reusing one persistent buffer across
# calls, overwritten each time, is safe under this codebase's existing single-threaded-per-inner-
# solve discipline (see guard_enter_inner_solve!/guard_exit_inner_solve!).
# ============================================================================

"Persistent, campaign-lifetime scratch for canonical_price_precompute's mu-dependent arrays. `logU` is a true campaign constant (log.(U) never changes); `mulU`/`UPow`/`UσPow` are recomputed (mu varies) but not reallocated."
mutable struct CanonicalPricePrecomputeWorkspace
    W::Int
    D::Int
    logU::Matrix{Float64}
    mulU::Matrix{Float64}
    UPow::Matrix{Float64}
    UσPow::Matrix{Float64}
    # 2026-08-11: mu used to fill mulU/UPow/UσPow, so canonical_price_precompute can skip refilling
    # them when mu has not moved. NaN sentinel => "never filled", and NaN != NaN makes the first
    # call always rebuild without a separate flag.
    #
    # WHY THIS IS SAFE, and why it is exactly as safe as what this workspace already does: mulU,
    # UPow and UσPow depend ONLY on (U, γ.Uσ, mu). This struct already treats U as campaign-constant
    # -- `logU` is filled once at construction and reused forever -- so keying the other three on mu
    # alone rests on precisely the same assumption that `logU` already rests on. If U could change
    # under a live workspace, `logU` would already be wrong today.
    #
    # It does NOT assume mu is fixed: when mu changes the arrays are rebuilt. That matters because
    # mu is campaign-constant only under the D=20 layout (context_real_d20.jl's
    # `free_idx = vcat(3 + Dact, Aod_offset+1:...)` covers gp and the A_od block, not mu = θ_full[1])
    # -- other layouts/modes may vary it, and they stay correct here.
    mu_filled::Float64
end

"One-time build: matches ctx.U's shape (W x D) and ctx.γ.Uσ's own shape for UσPow."
function build_canonical_price_precompute_workspace(ctx)
    U = ctx.U
    return CanonicalPricePrecomputeWorkspace(size(U, 1), ctx.D, log.(U), similar(U), similar(U),
                                             similar(ctx.γ.Uσ), NaN)
end

"""
    attach_canonical_price_precompute_workspace(ctx) -> ctx

Returns `ctx` merged with a `canonical_price_ws::CanonicalPricePrecomputeWorkspace` field, reusing
an existing matching-shape workspace (e.g. inherited via `reuse=`/a resumed checkpoint's
reconstructed context) rather than rebuilding, and rebuilding only on a genuine (W,D) shape
change -- same discipline as `attach_compressed_factual_workspace`. Call once per outer-solve
process.
"""
function attach_canonical_price_precompute_workspace(ctx)
    existing = hasproperty(ctx, :canonical_price_ws) ? ctx.canonical_price_ws : nothing
    W = size(ctx.U, 1); D = ctx.D
    ws = (existing isa CanonicalPricePrecomputeWorkspace && existing.W == W && existing.D == D) ?
        existing : build_canonical_price_precompute_workspace(ctx)
    return merge(ctx, (canonical_price_ws = ws,))
end
