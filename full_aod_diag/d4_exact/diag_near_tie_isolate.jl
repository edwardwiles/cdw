# Focused diagnostic: isolate why Section E of cm_cplus_expanded_d4_battery.jl shows a ~6e-2
# q0 discrepancy that does NOT shrink as the constructed near-tie separation shrinks toward the
# exact-tie boundary (eps 1e-2 down to 1e-8 all give ~6.04e-2) -- inconsistent with a genuine
# near-tie floating-point-noise explanation, which would predict recovery once eps exceeds each
# backend's own noise floor. Checks: (0) does a fresh REBUILD with UNMODIFIED U reproduce any
# discrepancy at all (rules out "any pcx/base rebuild is unsafe"); (1) do Reference's own winner0
# and C+'s own ref.winner disagree at the targeted (ω,d); (2) does the disagreement trace to a
# DIFFERENT (ω,d) than the one the tie was constructed at.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
using Printf, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
const L = 10
contrasts = :orthonormal
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts)
ctx_cm = pcx.ctx_cm; aug = pcx.aug; bins = pcx.bins; cctx = pcx.cctx
ws = build_lfix_factorized_workspace(D, W)
x_free_calib = ctx.θ0_up[ctx.free_idx]
base0, verify0 = archC_verified_state(x_free_calib, ctx_cm, cctx)
cache_ref0 = build_lfix_base_cache_cm(x_free_calib, ctx_cm, base0, ctx, aug, bins)
cache_cp0 = build_lfix_base_cache_cm_C!(ws, x_free_calib, ctx_cm, base0, ctx, aug, bins)
println("Step 0 (unmodified, single build, established path): max|q0_ref-q0_cp| = ", maximum(abs.(cache_ref0.q0 .- cache_cp0.q0)))

println()
println("Step 1: fresh REBUILD of pcx/base with UNMODIFIED ctx.U (isolates rebuild mechanics from any tie construction)")
pcx_fresh = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts)
base_fresh, _ = archC_verified_state(x_free_calib, pcx_fresh.ctx_cm, pcx_fresh.cctx)
ws_fresh = build_lfix_factorized_workspace(D, W)
cache_ref_fresh = build_lfix_base_cache_cm(x_free_calib, pcx_fresh.ctx_cm, base_fresh, ctx, pcx_fresh.aug, pcx_fresh.bins)
cache_cp_fresh = build_lfix_base_cache_cm_C!(ws_fresh, x_free_calib, pcx_fresh.ctx_cm, base_fresh, ctx, pcx_fresh.aug, pcx_fresh.bins)
maxerr_fresh = maximum(abs.(cache_ref_fresh.q0 .- cache_cp_fresh.q0))
println("  max|q0_ref_fresh - q0_cp_fresh| (Reference vs C+, both fresh-rebuilt, unmodified U) = ", maxerr_fresh)
println("  max|base_fresh.zeta* - base0.zeta*| = ", abs(base_fresh.ζstar - base0.ζstar))
println("  max|base_fresh.lambda* - base0.lambda*| = ", maximum(abs.(base_fresh.λstar .- base0.λstar)))
println("  max|q0_ref_fresh - q0_ref0| = ", maximum(abs.(cache_ref_fresh.q0 .- cache_ref0.q0)))

println()
println("Step 2: now apply the SAME near-tie U perturbation used in Section E (eps=1e-2, sign=+1) and repeat")
μ4 = base0.θ_full0[1]
γo4 = ctx.γ
Aod_θ4 = reshape(base0.θ_full0[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
lambda4 = reshape(γo4.P, (D, D))'
Aod4 = Aod_θ4 .* γo4.cHat .* (((γo4.wHat .* γo4.τ) ./ (γo4.wHat[1,1] .* γo4.τ[1,:]')) .^ (1/μ4)) .* (lambda4 ./ lambda4[1,:]')
AodPow4 = (Aod4 ./ γo4.cHat) .^ (-μ4)
constCons4 = [γo4.wHat[o] * AodPow4[o,d] * γo4.τ[o,d] for o in 1:D, d in 1:D]
s_tie, d_tie, o1, o2 = 1, 1, 1, 2
ratio = (constCons4[o1, d_tie] / constCons4[o2, d_tie])^(1/μ4)
Unear = copy(ctx.U)
Unear[s_tie, o2] = Unear[s_tie, o1] * ratio * 1.01
ctx_near = merge(ctx, (U = Unear,))
pcx_near = build_cm_production_context(ctx_near, CS; L = L, contrasts = contrasts)
base_near, _ = archC_verified_state(x_free_calib, pcx_near.ctx_cm, pcx_near.cctx)
ws_near = build_lfix_factorized_workspace(D, W)
cache_ref_near = build_lfix_base_cache_cm(x_free_calib, pcx_near.ctx_cm, base_near, ctx_near, pcx_near.aug, pcx_near.bins)
cache_cp_near = build_lfix_base_cache_cm_C!(ws_near, x_free_calib, pcx_near.ctx_cm, base_near, ctx_near, pcx_near.aug, pcx_near.bins)
maxerr_near = maximum(abs.(cache_ref_near.q0 .- cache_cp_near.q0))
idxw = argmax(abs.(cache_ref_near.q0 .- cache_cp_near.q0))
println("  max|q0_ref_near - q0_cp_near| = ", maxerr_near, "  worst idx=", idxw)
println("  q0_ref_near[worst]=", cache_ref_near.q0[idxw], "  q0_cp_near[worst]=", cache_cp_near.q0[idxw])
println("  Reference winner0[worst, d_tie=$d_tie] = ", cache_ref_near.winner0[idxw, d_tie])
println("  C+ ref.winner[worst, d_tie=$d_tie]      = ", cache_cp_near.ref.winner[idxw, d_tie])
println("  base_near.zeta* vs base0.zeta*: ", base_near.ζstar, " vs ", base0.ζstar)
println("  base_near.lambda* max diff from base0: ", maximum(abs.(base_near.λstar .- base0.λstar)))
println("  base_near nStatus: check inner solve status implicitly via no-throw (archC_verified_state)")

println()
println("Step 3: is the discrepancy located at d_tie=1 specifically, or spread elsewhere? Per-destination max|Δq0| contribution breakdown via contrib0")
maxerr_by_d = [maximum(abs.(cache_ref_near.contrib0[:, d] .- cache_cp_near.contrib0[:, d])) for d in 1:D]
println("  max|Δcontrib0| by destination: ", maxerr_by_d)
println("  cf_contrib0 max diff: ", maximum(abs.(cache_ref_near.cf_contrib0 .- cache_cp_near.cf_contrib0)))
println("DONE")

println()
println("Step 4: winner at d=2 (the ACTUAL discrepant destination, not d_tie=1) for ω=worst=1")
println("  Reference winner0[1, 2] = ", cache_ref_near.winner0[1, 2], "   price = ", cache_ref_near.winner_price0[1,2])
println("  Reference runnerup0[1, 2] = ", cache_ref_near.runnerup0[1, 2], "   price = ", cache_ref_near.runnerup_price0[1,2])
println("  C+ ref.winner[1, 2] = ", cache_cp_near.ref.winner[1, 2])
price_true, _, _ = factual_prices(base_near.θ_full0, ctx)
col = price_true[1, :, 2]
println("  TRUE price[ω=1, :, d=2] (direct factual_prices recompute) = ", col)
println("  argmin = ", argmin(col))

println()
println("Step 5: pTsigma reconstruction comparison at (omega=1, d=2, winner origin=2)")
println("  Reference dense pTσ0[1,2,2] = ", cache_ref_near.pTσ0[1, 2, 2])
sw_cp = cache_cp_near.ref.sw[1, 2]
println("  C+ log-score sw[1,2] = ", sw_cp, "  -> pTσ_from_score = ", pTσ_from_score(sw_cp, ctx.σ))
_, pTσ_true = price_and_pTsigma_cell(base_near.θ_full0, ctx, 2, 2)
println("  direct price_and_pTsigma_cell!(o=2,d=2) pTσ[ω=1] = ", pTσ_true[1])
println("  CONST_d[2] (should match between backends): ref=", cache_ref_near.CONST_d[2], "  cp=", cache_cp_near.CONST_d[2])
println("  lambda*[d1w for o=2,d=2] = ", base_near.λstar[2 + (2-1)*D])

println()
println("Step 6: FIX -- recompute gamma.Uσ consistently with Unear, retest")
γ_near_fixed = merge(ctx.γ, (Uσ = Unear .^ (1 .- ctx.σ),))
ctx_near_fixed = merge(ctx, (U = Unear, γ = γ_near_fixed))
pcx_near_fixed = build_cm_production_context(ctx_near_fixed, CS; L = L, contrasts = contrasts)
base_near_fixed, _ = archC_verified_state(x_free_calib, pcx_near_fixed.ctx_cm, pcx_near_fixed.cctx)
ws_near_fixed = build_lfix_factorized_workspace(D, W)
cache_ref_near_fixed = build_lfix_base_cache_cm(x_free_calib, pcx_near_fixed.ctx_cm, base_near_fixed, ctx_near_fixed, pcx_near_fixed.aug, pcx_near_fixed.bins)
cache_cp_near_fixed = build_lfix_base_cache_cm_C!(ws_near_fixed, x_free_calib, pcx_near_fixed.ctx_cm, base_near_fixed, ctx_near_fixed, pcx_near_fixed.aug, pcx_near_fixed.bins)
maxerr_fixed = maximum(abs.(cache_ref_near_fixed.q0 .- cache_cp_near_fixed.q0))
println("  max|q0_ref_near_fixed - q0_cp_near_fixed| (Uσ consistently recomputed) = ", maxerr_fixed)
println("  Reference pTσ0[1,2,2] fixed = ", cache_ref_near_fixed.pTσ0[1,2,2])
