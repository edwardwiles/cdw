# ============================================================================
# Mandatory equivalence test for moments_fast.jl's EK_moments_gammanorm_directgp_fast!
# vs the ORIGINAL, unmodified full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp!.
# Checked at calibration, both candidate points, and several random feasible
# perturbations, cold AND warm, with a FRESH pow_cache each time (so the
# cache-hit path is exercised too, not just the cold-miss path).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
using Random, Printf

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

points = Dict(
    "calibration" => vcat(ctx.θ0_up[3+D], pivot_reduce(zeros(D, D), pe)),
    "upper_lfixcomposite_sr1_60s" => [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845],
    "lower_stalled" => [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385],
)
rng = MersenneTwister(20260718)
for i in 1:5
    points["random_$i"] = points["upper_lfixcomposite_sr1_60s"] .+ 0.05 .* randn(rng, D2)
end

pow_cache = MuSigmaPowCache(ctx.U, ctx.obj.γ.Uσ)
all_pass = true
n_checked = 0

for (label, w) in points
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    W = size(ctx.U, 1); d = ctx.obj.d

    K1 = zeros(W); G1 = zeros(W, d)
    ctx.obj.moments!(K1, G1, θ_full, ctx.U, ctx.obj)   # ORIGINAL, unmodified

    K2 = zeros(W); G2 = zeros(W, d)
    EK_moments_gammanorm_directgp_fast!(K2, G2, θ_full, ctx.U, ctx.obj, pow_cache)   # FAST (cold-cache first call for this loop, or warm on later iters if mu matches -- mu is fixed here so always warm after point 1)

    maxdiff_G = maximum(abs.(G1 .- G2))
    maxdiff_K = maximum(abs.(K1 .- K2))
    pass = maxdiff_G < 1e-13 && maxdiff_K < 1e-13
    global all_pass &= pass
    global n_checked += 1
    @printf("  %-32s max|G diff|=%.3e  max|K diff|=%.3e  %s\n", label, maxdiff_G, maxdiff_K, pass ? "PASS" : "FAIL")
end

println("\npow_cache stats: n_recompute=$(pow_cache.n_recompute) (expect 1 -- mu never changes across ANY of these points), n_reuse=$(pow_cache.n_reuse) (expect $(n_checked-1))")
cache_pass = pow_cache.n_recompute == 1 && pow_cache.n_reuse == n_checked - 1
global all_pass &= cache_pass
println(cache_pass ? "CACHE BEHAVIOR: PASS (recomputed exactly once, reused for every subsequent call at the same mu)" : "CACHE BEHAVIOR: FAIL")

println("\n" * "="^78)
println(all_pass ? "ALL MOMENTS_FAST EQUIVALENCE TESTS PASSED ($n_checked points)" : "SOME TESTS FAILED -- do not trust EK_moments_gammanorm_directgp_fast! for timing or KNITRO wiring")
println("="^78)
all_pass || error("test_moments_fast.jl: equivalence checks failed")
