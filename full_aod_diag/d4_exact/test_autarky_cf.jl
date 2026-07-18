# ============================================================================
# Mandatory equivalence test: specialized focal-autarky CF path
# (autarky_cf.jl::EK_moments_gammanorm_directgp_autarkyCF!) vs the trusted
# production EK_moments_gammanorm_directgp! (full_aod_diag/moments_gammanorm.jl).
# The specialized path computes the SAME two scalars in the SAME float order and
# the SAME O(W) broadcast as hFunctionCounter!'s autarky branch, so output must
# be BIT-IDENTICAL (max|diff| == 0.0), a stronger claim than the 1e-13 the
# reassociation-based compressed path can make. Tested with pow_cache=nothing
# AND with a shared MuSigmaPowCache (compose check).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
include(joinpath(@__DIR__, "autarky_cf.jl"))
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
for i in 1:8
    points["random_$i"] = points["upper_lfixcomposite_sr1_60s"] .+ 0.05 .* randn(rng, D2)
end

pow_cache = MuSigmaPowCache(ctx.U, ctx.obj.γ.Uσ)
all_pass = true
n_checked = 0
worst_G = 0.0; worst_K = 0.0; worst_cf = 0.0

for (label, w) in points
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    W = size(ctx.U, 1); d = ctx.obj.d
    cf_col = D2 + 1

    K0 = zeros(W); G0 = zeros(W, d)
    EK_moments_gammanorm_directgp!(K0, G0, θ_full, ctx.U, ctx.obj)   # trusted production

    # specialized, no pow cache
    K1 = zeros(W); G1 = zeros(W, d)
    EK_moments_gammanorm_directgp_autarkyCF!(K1, G1, θ_full, ctx.U, ctx.obj)

    # specialized, shared pow cache
    K2 = zeros(W); G2 = zeros(W, d)
    EK_moments_gammanorm_directgp_autarkyCF!(K2, G2, θ_full, ctx.U, ctx.obj; pow_cache = pow_cache)

    dG1 = maximum(abs.(G0 .- G1)); dK1 = maximum(abs.(K0 .- K1))
    dG2 = maximum(abs.(G0 .- G2)); dK2 = maximum(abs.(K0 .- K2))
    dcf1 = maximum(abs.(G0[:, cf_col] .- G1[:, cf_col]))
    dcf2 = maximum(abs.(G0[:, cf_col] .- G2[:, cf_col]))
    pass = dG1 == 0.0 && dK1 == 0.0 && dG2 == 0.0 && dK2 == 0.0
    global all_pass &= pass
    global n_checked += 1
    global worst_G = max(worst_G, dG1, dG2); global worst_K = max(worst_K, dK1, dK2)
    global worst_cf = max(worst_cf, dcf1, dcf2)
    @printf("  %-30s |ΔG|=%.1e |ΔK|=%.1e |ΔCFcol|=%.1e (nocache)  |ΔG|=%.1e (cache)  %s\n",
            label, dG1, dK1, dcf1, dG2, pass ? "PASS" : "FAIL")
end

@printf("\nworst over %d points: |ΔG|=%.3e  |ΔK|=%.3e  |ΔCFcolumn|=%.3e\n", n_checked, worst_G, worst_K, worst_cf)
println("pow_cache: n_recompute=$(pow_cache.n_recompute) (expect 1), n_reuse=$(pow_cache.n_reuse)")
println("\n" * "="^78)
println(all_pass ? "ALL AUTARKY-CF EQUIVALENCE TESTS PASSED (bit-identical, max|diff|==0.0)" :
                   "SOME TESTS FAILED -- do not adopt the specialized autarky CF path")
println("="^78)
all_pass || error("test_autarky_cf.jl: equivalence checks failed")
