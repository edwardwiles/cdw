#!/usr/bin/env julia
# Line-level allocation-site audit (follow-up to perf_audit_allocations_2026-07-31.jl's
# @allocated findings) using Profile.Allocs, real D=20/W=80,000 data.
using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, LinearAlgebra, DelimitedFiles, Profile
using Profile.Allocs: @profile, fetch, clear
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

function report_top_allocs(label; topn=15)
    prof = fetch()
    println("\n--- $label: ", length(prof.allocs), " sampled allocations ---")
    # aggregate by (type, top-of-stack frame)
    counts = Dict{String,Tuple{Int,Int}}()
    for a in prof.allocs
        st = a.stacktrace
        frame = isempty(st) ? "?" : string(st[1].func, " @ ", basename(string(st[1].file)), ":", st[1].line)
        key = string(a.type) * " | " * frame
        (n, b) = get(counts, key, (0, 0))
        counts[key] = (n + 1, b + a.size)
    end
    rows = sort(collect(counts), by = kv -> -kv[2][2])
    for (k, (n, b)) in rows[1:min(topn, length(rows))]
        @printf("  %8d bytes  x%-4d  %s\n", b, n, k)
    end
end

# warmup everything first
melitz_update_operator_at_theta!(op, theta0, ctx)
melitz_update_operator_at_theta!(op, theta0, ctx)

println("=== melitz_update_operator_at_theta! ===")
clear(); Profile.Allocs.@profile sample_rate=1.0 melitz_update_operator_at_theta!(op, theta0, ctx)
report_top_allocs("melitz_update_operator_at_theta!")

n = obj20.outer_constr_index
x = zeros(n)
S = fill(1.0, op.W)
H = zeros(op.layout.num_moments + 1, op.layout.num_moments + 1)
melitz_full_weighted_gram_parallel!(H, op, S)
melitz_full_weighted_gram_parallel!(H, op, S)
println("\n=== melitz_full_weighted_gram_parallel! ===")
clear(); Profile.Allocs.@profile sample_rate=1.0 melitz_full_weighted_gram_parallel!(H, op, S)
report_top_allocs("melitz_full_weighted_gram_parallel!")

g_out = zeros(n)
obj20(x); obj20(x)
println("\n=== objective-only functor call ===")
clear(); Profile.Allocs.@profile sample_rate=1.0 obj20(x)
report_top_allocs("objective-only")

obj20(x, g_out); obj20(x, g_out)
println("\n=== objective+gradient functor call ===")
clear(); Profile.Allocs.@profile sample_rate=1.0 obj20(x, g_out)
report_top_allocs("objective+gradient")

h_out = zeros(n * (n + 1) ÷ 2)
obj20(x, Float64[]; h=h_out); obj20(x, Float64[]; h=h_out)
println("\n=== hessian-only functor call ===")
clear(); Profile.Allocs.@profile sample_rate=1.0 obj20(x, Float64[]; h=h_out)
report_top_allocs("hessian-only")

println("\nDONE.")
