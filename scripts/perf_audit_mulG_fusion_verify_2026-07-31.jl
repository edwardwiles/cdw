#!/usr/bin/env julia
# Verifies the mul_G! loop-fusion change (moment_operator.jl, 2026-07-31 perf audit) against
# a literal reimplementation of the ORIGINAL two-pass algorithm, on the same real D=20/
# W=80,000 operator state and several random dual vectors -- checks both exact bit-identity
# and, if not bit-identical, the actual max floating-point discrepancy (floating-point
# reassociation, (a+b)-c vs a+(b-c), is not guaranteed bit-identical in IEEE754).
using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, LinearAlgebra, DelimitedFiles, Random
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

REPO2 = dirname(@__DIR__)
function load_realD20_calib()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal
end
calib, focal = load_realD20_calib()
obj20, theta20 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=CappedEvaluation(10.0), outer_parameterization=:logcutoff)
ctx = obj20.γ
op = obj20.op

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(REPO2, "docs", "key_results", "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0 = theta_q_rows[("realD20_seed1_W80000", 0.5)]
melitz_update_operator_at_theta!(op, theta0, ctx)

# literal reimplementation of the ORIGINAL (pre-fusion) two-pass mul_G!
function mul_G_original!(u::AbstractVector{Float64}, op::MelitzMomentOperator, zeta::Real,
                          mu::AbstractVector{Float64})
    D = op.D; W = op.W
    zeta_f = Float64(zeta)
    @inbounds for s in 1:W
        u[s] = -zeta_f
    end
    trade_index = op.layout.trade_index
    coef = op.coef; lambda = op.lambda; order = op.order; bin = op.bin
    cum = zeros(D + 1)
    z_power = op.sorted_ctx.z_power_original
    @inbounds for o in 1:D
        const_o = 0.0
        cum[1] = 0.0
        for m in 1:D
            d = order[m, o]
            mu_od = mu[trade_index[o, d]]
            cum[m+1] = cum[m] + mu_od * coef[o, d]
            const_o += mu_od * lambda[o, d]
        end
        for s in 1:W
            u[s] += const_o
        end
        for s in 1:W
            b = bin[s, o]
            b == 0 && continue
            u[s] -= z_power[s, o] * cum[b+1]
        end
    end
    mu_link = mu[op.layout.focal_link_index]
    if mu_link != 0.0
        ell = op.ell
        @inbounds for s in 1:W
            u[s] -= mu_link * ell[s]
        end
    end
    return u
end

rng = MersenneTwister(2026)
n = obj20.outer_constr_index
u_new = zeros(op.W)
u_old = zeros(op.W)
worst_maxabs = 0.0
worst_maxrel = 0.0
n_trials = 20
for t in 1:n_trials
    x = randn(rng, n) .* (t == 1 ? 0.0 : 1.0)   # trial 1: all-zero mu (edge case); rest: random
    zeta = x[1]; mu = x[2:end]
    mul_G!(u_new, op, zeta, mu)
    mul_G_original!(u_old, op, zeta, mu)
    d = maximum(abs, u_new .- u_old)
    identical = u_new == u_old
    rel = d / max(1e-300, maximum(abs, u_old))
    global worst_maxabs = max(worst_maxabs, d)
    global worst_maxrel = max(worst_maxrel, rel)
    @printf("trial %2d: bit-identical=%-5s  max|Δ|=%.3e  max relative=%.3e\n", t, identical, d, rel)
end
@printf("\nWORST over %d trials: max|Δ|=%.3e  max relative=%.3e\n", n_trials, worst_maxabs, worst_maxrel)

println("\n=== wall-clock: fused vs original, real D=20/W=80,000, 500 calls each ===")
x = randn(rng, n)
zeta = x[1]; mu = x[2:end]
mul_G!(u_new, op, zeta, mu); mul_G_original!(u_old, op, zeta, mu)  # warmup
t_new = @elapsed for _ in 1:500; mul_G!(u_new, op, zeta, mu); end
t_old = @elapsed for _ in 1:500; mul_G_original!(u_old, op, zeta, mu); end
@printf("fused:    %.6f s/call\n", t_new/500)
@printf("original: %.6f s/call\n", t_old/500)
@printf("speedup:  %.2fx\n", (t_old/500)/(t_new/500))
