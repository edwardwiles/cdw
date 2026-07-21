# Continuation 14, Task 3 fix verification: price_and_pTsigma_cell! (lfix_incremental.jl) must
# produce output BIT-FOR-BIT identical to the original allocating price_and_pTsigma_cell, and
# build_lfix_base_cache (now wired to the in-place variant) must still work correctly end-to-end,
# at 2 real points, with a real allocation reduction confirmed (not just assumed).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
using Printf, LinearAlgebra, Random, Dates

lp(xs...) = (println(xs...); flush(stdout))
lp("=== c14_verify_lfix_buffer_fix === ", Dates.now())
Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs", time()-t0))

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx.θ0_up[3+D]
xf_calib = x_free_from_w2(vcat(gp0 * 1.01, zfree0))

Random.seed!(777)
dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
xf_near = x_free_from_w2(vcat(gp0 * 1.01, zfree0 .+ 0.05 .* dir))

points = [("calibration", xf_calib), ("nearby_perturbed", xf_near)]

all_ok = true
for (label, xf) in points
    lp(""); lp("-"^90); lp("POINT: ", label); lp("-"^90)
    θ_full = CS.reconstruct_full(xf, ctx.m)

    # ---- direct bit-for-bit check of the two price_and_pTsigma_cell variants ----
    W = size(ctx.obj.U, 1)
    price_ref = Array{Float64}(undef, W, D, D); pTσ_ref = Array{Float64}(undef, W, D, D)
    price_new = Array{Float64}(undef, W, D, D); pTσ_new = Array{Float64}(undef, W, D, D)
    for d in 1:D, o in 1:D
        p, ps = price_and_pTsigma_cell(θ_full, ctx, o, d)
        price_ref[:, o, d] .= p; pTσ_ref[:, o, d] .= ps
        price_and_pTsigma_cell!(@view(price_new[:, o, d]), @view(pTσ_new[:, o, d]), θ_full, ctx, o, d)
    end
    exact_price = price_ref == price_new
    exact_pTσ = pTσ_ref == pTσ_new
    lp("  price0 bit-for-bit identical: ", exact_price, "   max_abs_diff=", maximum(abs.(price_ref .- price_new)))
    lp("  pTσ0  bit-for-bit identical: ", exact_pTσ, "   max_abs_diff=", maximum(abs.(pTσ_ref .- pTσ_new)))
    global all_ok &= exact_price && exact_pTσ

    # ---- end-to-end build_lfix_base_cache (now wired to the fix), with allocation measurement ----
    base = solve_base_state(xf, ctx)
    GC.gc()
    stats = @timed build_lfix_base_cache(xf, ctx, base; validate_dense = false)
    cache = stats.value
    lp(@sprintf("  build_lfix_base_cache: wall=%.3fs  bytes=%.3e (%.1f MB)  [baseline pre-fix was ~1078MB at calibration]",
        stats.time, stats.bytes, stats.bytes/2^20))
    lp("  q0 finite: ", all(isfinite, cache.q0), "  length(q0)=", length(cache.q0))

    # ---- self-validation cross-check: validate_dense=true's OWN independent dense rebuild must
    #      still agree with the (now differently-computed) cache contents -- this is the STRONGEST
    #      available correctness gate, already built into this codebase, reused not reinvented ----
    stats_val = @timed build_lfix_base_cache(xf, ctx, base; validate_dense = true)
    lp("  validate_dense=true self-check: ", stats_val.value !== nothing ? "PASSED (no error thrown)" : "?")
end

lp(""); lp("="^90)
lp(all_ok ? "ALL POINTS: price_and_pTsigma_cell! bit-for-bit identical to the original -- FIX VERIFIED SAFE" :
            "MISMATCH FOUND -- DO NOT TRUST THE FIX, investigate before relying on it")
lp("DONE_C14_VERIFY_LFIX_BUFFER_FIX")
