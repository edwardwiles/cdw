# Quick, cheap, isolated-process check: does the data-anchored ACR/Chaney sufficient-
# statistic cross-check (acr_gains_from_trade, equilibrium.jl) agree with the reported
# gamma_prime-driven GT at the two final D20 profiled-A incumbents (upper GT=8.849341%,
# lower GT=0.272266%)? This does NOT call melitz_moments!/mul_G!/mul_Gt! -- it only loads
# the fresh calibration + the two checkpoints and evaluates the closed-form GT formulas.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization
LinearAlgebra.BLAS.set_num_threads(1)

function load_realD20_calib()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal, countries
end
calib, focal, countries_d20 = load_realD20_calib()
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx = obj_d20.γ
D = ctx.D
j = ctx.target_country
@printf("D=%d  target_country=%d (%s)  sigma=%.4f  theta_star=%.6f\n", D, j, countries_d20[j], ctx.sigma, ctx.theta_star)
@printf("X_data[j,j]=%.10g  expenditure[j]=%.10g  lambda_jj(data)=%.10g\n",
    ctx.X_data[j, j], ctx.expenditure[j], ctx.X_data[j, j] / ctx.expenditure[j])
@printf("w[j]=%.10g  w_prime=%.10g\n", ctx.w[j], ctx.w_prime)

mutable struct ChainPoint
    idx::Int; phase::Symbol; g::Float64; GT::Float64; classification::Symbol; Delta::Float64
    within_budget::Bool; lfd_ok::Bool; lfd_Delta::Float64; A_free::Vector{Float64}; theta_free::Vector{Float64}
    f_full::Matrix{Float64}; q_full::Matrix{Float64}; lfd_weights::Vector{Float64}; nStatus::Int
    unique_inner_solves::Int; wall_s::Float64; timestamp::String
end
mutable struct ChainState
    chain_label::String; direction::Symbol; points::Vector{ChainPoint}
    bracket::Union{Nothing,NTuple{4,Float64}}; n_evaluated::Int; most_extreme_idx::Int
    n_nonimproving::Int; elapsed_wall_s::Float64; status::Symbol
    cur_A_free::Vector{Float64}; cur_q::Matrix{Float64}; cur_p_star::Vector{Float64}
    cur_g::Float64; cur_GT::Float64; step_gt::Float64; n_accepted::Int
end

OVERDIR = joinpath(REPO2, "docs", "key_results", "overnight_qpoll_delta0p5_2026-07-31")
for (label, direction) in [("phase6_upper_m1", "upper"), ("phase6_lower_m-1", "lower")]
    ckpt = joinpath(OVERDIR, "phase5", "checkpoints", "$(label).jls")
    state = deserialize(ckpt)
    best = state.points[state.most_extreme_idx]
    gamma_prime_j = exp(best.g)
    wage_ratio = ctx.w_prime / ctx.w[j]
    kappa_ratio = wage_ratio * gamma_prime_j^(1 / (ctx.sigma - 1))
    GT_model = 100 * (1 - kappa_ratio)
    lambda_jj = ctx.X_data[j, j] / ctx.expenditure[j]
    GT_ACR = 100 * (1 - lambda_jj^(1 / ctx.theta_star))
    println("\n", "="^90)
    @printf("[%s / %s]  reported GT (checkpointed) = %.6f%%   g=log(gamma_prime_j)=%.6f  gamma_prime_j=%.6g\n",
        label, direction, best.GT, best.g, gamma_prime_j)
    @printf("  GT_model recomputed from (gamma_prime_j, wage_ratio)      = %.6f%%   (match to checkpoint: %.3e)\n",
        GT_model, abs(GT_model - best.GT))
    @printf("  GT_ACR from DATA-anchored domestic trade share lambda_jj  = %.6f%%   (lambda_jj=%.6g)\n",
        GT_ACR, lambda_jj)
    @printf("  |GT_model - GT_ACR| = %.6f percentage points\n", abs(GT_model - GT_ACR))
end
