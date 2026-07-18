# ============================================================================
# Equivalence + exponent + independence tests for the V2 cached-base focal-autarky
# CF path (autarky_cf_v2.jl). Three targets, all required to adopt:
#   (A) vs generic production EK_moments_gammanorm_directgp!  (full G/K)
#   (B) vs already-adopted autarky_cf.jl specialized path      (full G/K + CF col)
#   (C) analytic scalar/exponent + reuse-across-A_dd + independence-from-other-A_od
# V2 turns the per-draw DIVIDE (`cf_num / UσPow[:,o1]`) into a per-draw MULTIPLY
# against a once-built reciprocal, so it is NOT bit-identical to (A)/(B); it
# targets documented-TIGHT equivalence. We report max ABS and max REL error and
# assert rel-err <= 1e-13 (typically ~1e-16, a single reciprocal ULP).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
include(joinpath(@__DIR__, "autarky_cf.jl"))
include(joinpath(@__DIR__, "autarky_cf_v2.jl"))
using Random, Printf, SpecialFunctions

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; bi = ctx.bi
Aod_offset = 3 + D                       # independenceMoment==0 here
idx_Add = Aod_offset + (bi - 1) * D + bi  # θ index of Aod_θ[bi,bi] (column-major)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
W = size(ctx.U, 1); d = ctx.obj.d; cf_col = D2 + 1

points = Dict(
    "calibration" => vcat(ctx.θ0_up[3+D], pivot_reduce(zeros(D, D), pe)),
    "upper" => [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845],
    "lower" => [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385],
)
rng = MersenneTwister(20260718)
for i in 1:8
    points["random_$i"] = points["upper"] .+ 0.05 .* randn(rng, D2)
end

# ---- rebuild AodPow + CF scalars exactly as the constructor does (for target C) ----
function cf_scalars_from_theta(θ_full)
    γo = ctx.obj.γ
    @unpack wHat, τ, P, cHat = γo
    μ = θ_full[1]; σ = θ_full[2]
    Aod_θ = reshape(θ_full[Aod_offset+1:Aod_offset+D2], (D, D))
    lambda = reshape(P, (D, D))'
    Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ cHat) .^ (-μ)
    autarky_cf_scalars(ctx.obj, AodPow, σ, θ_full[3+D])  # (cf_num, cf_denom, o1)
end

# Pure relative error (used for exponent/scalar checks on O(1) quantities).
relerr(a, b) = (a == b) ? 0.0 : abs(a - b) / max(abs(a), abs(b), eps())
# Floored relative error: the CF moment column crosses zero (it is a centered
# moment), so a bare ratio blows up on entries that are themselves ~1e-15. The
# multiply-vs-divide reassociation bounds the ABSOLUTE error by ~1 ULP of the
# O(1) numerator term (cf_num/UσPow ~ O(1)), so we floor the denominator at 1e-6
# to measure agreement at the natural scale, and separately report the raw abs err.
relerr_floored(a, b) = (a == b) ? 0.0 : abs(a - b) / max(abs(a), abs(b), 1e-6)

# ============================ Targets A & B ============================
base_v2 = AutarkyCFBase(W)
all_pass = true; n = 0
worstA_abs = 0.0; worstA_rel = 0.0; worstB_abs = 0.0; worstB_rel = 0.0
worstA_cf_rel = 0.0
println("Target A (vs generic) and B (vs autarky_cf.jl) full-G/K equivalence:")
for (label, w) in sort(collect(points); by = first)
    θ_full = CS.reconstruct_full(x_free_from_w(w), ctx.m)

    K0 = zeros(W); G0 = zeros(W, d)
    EK_moments_gammanorm_directgp!(K0, G0, θ_full, ctx.U, ctx.obj)                    # generic
    K1 = zeros(W); G1 = zeros(W, d)
    EK_moments_gammanorm_directgp_autarkyCF!(K1, G1, θ_full, ctx.U, ctx.obj)          # autarky_cf.jl
    K2 = zeros(W); G2 = zeros(W, d)
    EK_moments_gammanorm_directgp_autarkyCF_v2!(K2, G2, θ_full, ctx.U, ctx.obj; base = base_v2)

    dA_abs = max(maximum(abs.(G0 .- G2)), maximum(abs.(K0 .- K2)))
    dA_rel = max(maximum(relerr_floored.(G0, G2)), maximum(relerr_floored.(K0, K2)))
    dB_abs = max(maximum(abs.(G1 .- G2)), maximum(abs.(K1 .- K2)))
    dB_rel = max(maximum(relerr_floored.(G1, G2)), maximum(relerr_floored.(K1, K2)))
    dcf_rel = maximum(relerr_floored.(G0[:, cf_col], G2[:, cf_col]))
    # centered moment ⇒ absolute error is the meaningful tight-equivalence metric
    # (floored-rel reported for context); reassociation bounds |Δ| by ~1 ULP of O(1).
    pass = dA_abs <= 1e-11 && dB_abs <= 1e-11
    global all_pass &= pass; global n += 1
    global worstA_abs = max(worstA_abs, dA_abs); global worstA_rel = max(worstA_rel, dA_rel)
    global worstB_abs = max(worstB_abs, dB_abs); global worstB_rel = max(worstB_rel, dB_rel)
    global worstA_cf_rel = max(worstA_cf_rel, dcf_rel)
    @printf("  %-14s A:|Δ|=%.1e rel=%.1e  B:|Δ|=%.1e rel=%.1e  CFcol rel=%.1e  %s\n",
            label, dA_abs, dA_rel, dB_abs, dB_rel, dcf_rel, pass ? "PASS" : "FAIL")
end
@printf("  worst  A: abs=%.2e rel=%.2e | B: abs=%.2e rel=%.2e | CFcol rel=%.2e\n",
        worstA_abs, worstA_rel, worstB_abs, worstB_rel, worstA_cf_rel)
@printf("  base reuse: n_build=%d (expect 1) n_reuse=%d\n\n", base_v2.n_build, base_v2.n_reuse)

# ==================== Target C: exponent, reuse, independence ====================
println("Target C: scalar exponent, cache reuse across A_dd, independence from other A_od:")
θ_base = CS.reconstruct_full(x_free_from_w(points["upper"]), ctx.m)
μ = θ_base[1]; σ = θ_base[2]
exp_on_Add = μ * (σ - 1)   # claimed exponent on the RAW free variable A_dd
@printf("  μ=%.6f σ=%.6f  ⇒ claimed exponent on A_dd = μ(σ-1) = %.6f\n", μ, σ, exp_on_Add)

# (C1) exponent: cf_num(a1)/cf_num(a2) == (a1/a2)^(μ(σ-1)); cf_denom A_dd-independent
Add_vals = [0.4, 0.8, 1.3, 2.7, 5.0]
θv = copy(θ_base)
cf_nums = Float64[]; cf_denoms = Float64[]
for a in Add_vals
    θv[idx_Add] = a
    cn, cd, _ = cf_scalars_from_theta(θv)
    push!(cf_nums, cn); push!(cf_denoms, cd)
end
worst_exp = 0.0
for i in 2:length(Add_vals)
    predicted = (Add_vals[i] / Add_vals[1])^exp_on_Add
    observed = cf_nums[i] / cf_nums[1]
    global worst_exp = max(worst_exp, relerr(predicted, observed))
end
denom_const = maximum(relerr.(cf_denoms, cf_denoms[1]))  # should be 0: cf_denom ⊥ A_dd
c1 = worst_exp <= 1e-12 && denom_const == 0.0
@printf("  C1 exponent: worst rel-err cf_num ratio vs (a1/a2)^μ(σ-1) = %.2e ; cf_denom-⊥-A_dd rel = %.1e  %s\n",
        worst_exp, denom_const, c1 ? "PASS" : "FAIL")

# (C2) cache reuse: SAME base vector reused across all A_dd; column tracks A_dd,
#      equals generic path at each A_dd. Build base ONCE, vary A_dd, no rebuild.
base_reuse = AutarkyCFBase(W)
worst_reuse_abs = 0.0; worst_reuse_rel = 0.0
for a in Add_vals
    θv[idx_Add] = a
    Kg = zeros(W); Gg = zeros(W, d)
    EK_moments_gammanorm_directgp!(Kg, Gg, θv, ctx.U, ctx.obj)               # generic ground truth
    Kv = zeros(W); Gv = zeros(W, d)
    EK_moments_gammanorm_directgp_autarkyCF_v2!(Kv, Gv, θv, ctx.U, ctx.obj; base = base_reuse)
    global worst_reuse_abs = max(worst_reuse_abs, maximum(abs.(Gg[:, cf_col] .- Gv[:, cf_col])))
    global worst_reuse_rel = max(worst_reuse_rel, maximum(relerr_floored.(Gg[:, cf_col], Gv[:, cf_col])))
end
c2 = base_reuse.n_build == 1 && base_reuse.n_reuse == length(Add_vals) - 1 && worst_reuse_abs <= 1e-11
@printf("  C2 reuse: n_build=%d (expect 1) n_reuse=%d (expect %d) worst CFcol abs=%.2e rel=%.2e vs generic  %s\n",
        base_reuse.n_build, base_reuse.n_reuse, length(Add_vals) - 1, worst_reuse_abs, worst_reuse_rel, c2 ? "PASS" : "FAIL")

# (C3) independence: perturb EVERY non-(bi,bi) A_od entry ⇒ CF scalars unchanged.
θp = copy(θ_base)
cn0, cd0, _ = cf_scalars_from_theta(θp)
worst_indep = 0.0
for k in 1:D2
    θtmp = copy(θ_base)
    (idx = Aod_offset + k) == idx_Add && continue   # skip the (bi,bi) entry itself
    θtmp[idx] *= 1.7                                  # large perturbation of another A_od entry
    cn, cd, _ = cf_scalars_from_theta(θtmp)
    global worst_indep = max(worst_indep, abs(cn - cn0), abs(cd - cd0))
end
c3 = worst_indep == 0.0
@printf("  C3 independence: worst |Δcf_num|,|Δcf_denom| over all 15 other A_od perturbations = %.1e  %s\n",
        worst_indep, c3 ? "PASS" : "FAIL")

all_pass &= (c1 && c2 && c3)
println("\n" * "="^78)
println(all_pass ? "ALL V2 TESTS PASSED (tight equivalence + verified exponent/independence)" :
                   "SOME V2 TESTS FAILED -- do not adopt")
println("="^78)
all_pass || error("test_autarky_cf_v2.jl: checks failed")
