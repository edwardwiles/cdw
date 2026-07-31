#!/usr/bin/env julia
# Allocation audit for moment construction + inner callback eval/gradient/hessian
# (2026-07-31 perf follow-up to the legacy-H audit). Measures @allocated on the REAL
# production functions after warmup (JIT-compiled), at real D=20/W=80,000 production data.
using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, LinearAlgebra, DelimitedFiles
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

println("Julia threads = ", Threads.nthreads(), "   BLAS threads = ", LinearAlgebra.BLAS.get_num_threads())

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
D = ctx.D
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

println("\n=== moment/operator update: melitz_update_operator_at_theta! (once-per-outer-point, NOT claimed zero-alloc) ===")
melitz_update_operator_at_theta!(op, theta0, ctx)  # warmup
b1 = @allocated melitz_update_operator_at_theta!(op, theta0, ctx)
b2 = @allocated melitz_update_operator_at_theta!(op, theta0, ctx)
@printf("  bytes: warm2=%d warm3=%d\n", b1, b2)

println("\n=== mul_G! / mul_Gt! (claimed zero-alloc) ===")
n = obj20.outer_constr_index
x = copy(obj20.x)
if !all(isfinite, x)
    x = zeros(n)
end
zeta = x[1]; mu = x[2:end]
u = zeros(op.W)
mul_G!(u, op, zeta, mu)  # warmup
bG1 = @allocated mul_G!(u, op, zeta, mu)
bG2 = @allocated mul_G!(u, op, zeta, mu)
@printf("  mul_G! bytes: warm2=%d warm3=%d\n", bG1, bG2)

g = zeros(op.layout.num_moments)
mul_Gt!(g, op, u)
bGt1 = @allocated mul_Gt!(g, op, u)
bGt2 = @allocated mul_Gt!(g, op, u)
@printf("  mul_Gt! bytes: warm2=%d warm3=%d\n", bGt1, bGt2)

println("\n=== melitz_full_weighted_gram! (serial) / _parallel (claimed zero-alloc) ===")
S = abs.(u) .+ 1.0   # arbitrary positive weight vector, same shape ddPsi! would produce
H = zeros(op.layout.num_moments + 1, op.layout.num_moments + 1)
melitz_full_weighted_gram!(H, op, S)
bH1 = @allocated melitz_full_weighted_gram!(H, op, S)
bH2 = @allocated melitz_full_weighted_gram!(H, op, S)
@printf("  melitz_full_weighted_gram! (serial) bytes: warm2=%d warm3=%d\n", bH1, bH2)

melitz_full_weighted_gram_parallel!(H, op, S)
bHp1 = @allocated melitz_full_weighted_gram_parallel!(H, op, S)
bHp2 = @allocated melitz_full_weighted_gram_parallel!(H, op, S)
@printf("  melitz_full_weighted_gram_parallel! bytes: warm2=%d warm3=%d\n", bHp1, bHp2)

println("\n=== MelitzCCBundle functor: objective-only / objective+gradient / hessian-only (claimed zero-alloc after warmup) ===")
melitz_update_operator_at_theta!(op, theta0, ctx)
g_out = zeros(n)
h_out = zeros(n * (n + 1) ÷ 2)

obj20(x)  # warmup obj-only
bO1 = @allocated obj20(x)
bO2 = @allocated obj20(x)
@printf("  objective-only bytes: warm2=%d warm3=%d\n", bO1, bO2)

obj20(x, g_out)  # warmup obj+grad
bOG1 = @allocated obj20(x, g_out)
bOG2 = @allocated obj20(x, g_out)
@printf("  objective+gradient bytes: warm2=%d warm3=%d\n", bOG1, bOG2)

obj20(x, Float64[]; h=h_out)  # warmup hessian-only
bH_1 = @allocated obj20(x, Float64[]; h=h_out)
bH_2 = @allocated obj20(x, Float64[]; h=h_out)
@printf("  hessian-only bytes: warm2=%d warm3=%d\n", bH_1, bH_2)

println("\n=== structured Hessian: :structured_serial vs :structured_parallel field wall-time, D=", D, " W=", op.W, " ===")
melitz_full_weighted_gram!(H, op, S)  # ensure warm
t1 = @elapsed for _ in 1:20; melitz_full_weighted_gram!(H, op, S); end
melitz_full_weighted_gram_parallel!(H, op, S)
t2 = @elapsed for _ in 1:20; melitz_full_weighted_gram_parallel!(H, op, S); end
@printf("  serial: %.4f s / call   parallel: %.4f s / call  (nthreads=%d)\n", t1/20, t2/20, Threads.nthreads())

println("\n=== mul_G! wall-time (per call) ===")
t3 = @elapsed for _ in 1:200; mul_G!(u, op, zeta, mu); end
@printf("  mul_G!: %.6f s / call\n", t3/200)
t4 = @elapsed for _ in 1:200; mul_Gt!(g, op, u); end
@printf("  mul_Gt!: %.6f s / call\n", t4/200)

println("\nDONE.")
