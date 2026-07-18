# ============================================================================
# Continuation 8, Section 9 (nested-W). Builds ONE common W=80000 draw pool
# with a FIXED seed and provides `build_nested_ctx(W; ...)` to construct a
# D=4 exact context whose draws are an EXACT ROW-PREFIX of that pool -- i.e.
# genuinely nested draws across W=8000/20000/80000, not three independent
# draw sets (the task's explicit requirement: "independent draw sets are not
# a stability experiment").
#
# WHY NOT JUST CALL d_exact_setup_scaled(D=4,W=W) THREE TIMES:
# docs/fullA_d4_W_stability.md (an earlier, separate investigation in this
# repo) already flagged that context_scaled.jl's own internal drawU() call
# does NOT nest across W: prepare_cc/drawU.jl -> genRands.jl calls
# `rand!(U)` on a freshly zeros(W,D)-allocated matrix, which fills in
# COLUMN-MAJOR linear order. After `Random.seed!(seedU)`, column 1 (RNG
# stream positions 1:W) IS the same regardless of W, but column 2 starts at
# stream position W+1 -- which DIFFERS between e.g. W=8000 (starts at 8001)
# and W=80000 (starts at 80001). So only column 1 of 4 would coincidentally
# nest; columns 2-4 would not. Verified by reading the source, not assumed.
#
# THE FIX: draw ONE big Wmax x D pool ourselves, ONCE, with the same
# elementwise transform genExpRands! uses (rand! then -log(1-x)). Nesting
# then follows trivially: NESTEDW_POOL[1:W, :] for different W are literal
# row-prefixes of one already-materialized array -- no RNG-stream-alignment
# reasoning required at all.
#
# WIRING THE POOL IN: build a context the NORMAL way via
# context_scaled.jl::d_exact_setup_scaled (so every internal buffer --
# obj.H, obj.arg0/1/2, obj.jac_h, etc. -- is correctly shaped for the target
# W), then OVERWRITE its draws in place with NESTEDW_POOL[1:W,:]. Three
# arrays hold draws and must all be kept consistent:
#   - obj.U            (read directly by oracle.jl's evaluate_fullA: `obj.moments!(K,G,θ,obj.U,obj)`)
#   - obj.γ.Ū          (= copy(U), per prepare_cc/createUDerivatives!.jl)
#   - obj.γ.Uσ         (= U .^ (1-σHat), per prepare_cc/createUDerivatives!.jl)
# Both derived arrays are SIMPLE ELEMENTWISE functions of U (verified by
# reading createUDerivatives!.jl -- no cross-row dependence, so recomputing
# them locally from the swapped-in U is exact, not an approximation).
# ctx.U / ctx.pp.U / ctx.γ / ctx.obj.γ are the SAME underlying objects as
# obj.U / obj.γ (context_scaled.jl never copies when threading pp's fields
# into the PsiObjectiveBundleImplicit constructor) -- so mutating obj.U and
# obj.γ.Ū/Uσ in place with `.=` already updates every alias. All three are
# still set/asserted explicitly below for clarity, not left to silent
# aliasing.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
using Random

# Deliberately DISTINCT from AD_PARAMS.seedU=888 -- that seed is baked into
# the draw set that ORIGINALLY FOUND the registered candidates (via
# d4_exact_setup()/d_exact_setup_scaled's own internal drawU()). This
# workstream's pool is a SEPARATE, fresh resampling draw used only to probe
# W-sensitivity of a fixed (g,A) point and to re-optimize against more/fewer
# draws -- not a replay of the discovery draws. Documented, not accidental.
const NESTEDW_SEED = 91234
const NESTEDW_WMAX = 80_000
const NESTEDW_D = 4

"""
    build_nestedw_pool(; D=4, Wmax=80000, seed=91234) -> Matrix{Float64} (Wmax x D)

Draws the ONE shared exp(1) pool this whole workstream nests prefixes out of.
Identical elementwise transform to prepare_cc/genRands.jl::genExpRands!
(`rand!` then `-log(1-x)`), called on the global RNG after an explicit seed.
"""
function build_nestedw_pool(; D::Int = NESTEDW_D, Wmax::Int = NESTEDW_WMAX, seed::Int = NESTEDW_SEED)
    Random.seed!(seed)
    U = zeros(Wmax, D)
    rand!(U)
    @. U = -log(1 - U)
    return U
end

const NESTEDW_POOL = build_nestedw_pool()

"""
    build_nested_ctx(W; find_smallest=true, outer_loop_opt=..., inner_loop_opt=...) -> ctx

Like `d_exact_setup_scaled(D=4, W=W)`, but with the draws (obj.U, obj.γ.Ū,
obj.γ.Uσ) overwritten in place to be EXACTLY `NESTEDW_POOL[1:W, :]` (and its
derived arrays) -- a genuine row-prefix of the single shared pool, so
W=8000/20000/80000 runs use literally the same underlying randomness up to
truncation.
"""
function build_nested_ctx(W::Int; find_smallest::Bool = true,
        outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
        inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"))
    W <= NESTEDW_WMAX || error("build_nested_ctx: W=$W exceeds pool size $NESTEDW_WMAX")
    ctx = d_exact_setup_scaled(D = NESTEDW_D, W = W, find_smallest = find_smallest,
                                outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt)
    Uw = NESTEDW_POOL[1:W, :]
    σ = ctx.σ
    size(ctx.obj.U) == size(Uw) ||
        error("build_nested_ctx: shape mismatch ctx.obj.U=$(size(ctx.obj.U)) vs pool prefix=$(size(Uw))")
    ctx.obj.U .= Uw
    ctx.U .= Uw
    ctx.pp.U .= Uw
    ctx.obj.γ.Ū .= Uw
    ctx.obj.γ.Uσ .= Uw .^ (1 - σ)
    # self-consistency assertions (cheap, always run -- catches a wiring regression immediately
    # rather than silently producing wrong numbers downstream)
    @assert ctx.obj.U == Uw
    @assert ctx.obj.γ.Ū == Uw
    @assert ctx.obj.γ.Uσ ≈ Uw .^ (1 - σ)
    @assert ctx.obj.U === ctx.U === ctx.pp.U   # confirm the aliasing this whole design leans on
    return ctx
end
