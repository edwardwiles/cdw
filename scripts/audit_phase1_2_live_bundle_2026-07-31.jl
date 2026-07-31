#!/usr/bin/env julia
# Phase 1/2 live-object + counter audit for the Melitz legacy-H removal task
# (docs/melitz_legacy_H_removal_audit_2026-07-31.md). Constructs the bundle EXACTLY the way
# real production code does -- no backend override, no cc_algo/CounterfactualSensitivity
# loaded (mirroring scripts/melitz_w_sensitivity_diagnostics_2026-07-28.jl's own setup) -- and
# inspects/instruments the live object rather than trusting any docstring claim.
#
# Usage: julia --project=. -t 20 scripts/audit_phase1_2_live_bundle_2026-07-31.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, LinearAlgebra, DelimitedFiles
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

println("="^90)
println("Julia threads = ", Threads.nthreads(), "   BLAS threads = ", LinearAlgebra.BLAS.get_num_threads())
println("="^90)

function rss_mb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                return parse(Float64, split(line)[2]) / 1024
            end
        end
    catch
    end
    return NaN
end

melitz_backend_counters_reset!()

# ---------------------------------------------------------------------------
# D=4 fixture (memory-confirmed non-fragile seed 29), production-default bundle construction
# ---------------------------------------------------------------------------
println("\n--- D=4: build_melitz_psi_bundle with PURE PRODUCTION DEFAULTS (no backend kwarg) ---")
data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
rss0 = rss_mb()
obj4, theta4 = build_melitz_psi_bundle(data4; policy=CappedEvaluation(10.0))
rss1 = rss_mb()

println("concrete bundle type      = ", typeof(obj4))
println("fieldnames                = ", fieldnames(typeof(obj4)))
has_H = hasproperty(obj4, :H) || :H in fieldnames(typeof(obj4))
has_G = hasproperty(obj4, :G) || :G in fieldnames(typeof(obj4))
has_K = hasproperty(obj4, :K) || :K in fieldnames(typeof(obj4))
has_moments = hasproperty(obj4, :moments!) || :moments! in fieldnames(typeof(obj4))
println("has .H field?              = ", has_H)
println("has .G field?              = ", has_G)
println("has .K field?              = ", has_K)
println("has .moments! field?       = ", has_moments)
println("RSS before/after construct = ", round(rss0,digits=1), " / ", round(rss1,digits=1), " MB  (delta=", round(rss1-rss0,digits=2), ")")

println("\n-- field sizes / bytes --")
for fld in fieldnames(typeof(obj4))
    v = getfield(obj4, fld)
    if v isa AbstractArray
        @printf("  %-28s %-20s size=%-14s bytes=%d\n", fld, typeof(v), string(size(v)), Base.summarysize(v))
    end
end
println("  TOTAL Base.summarysize(obj4) bytes = ", Base.summarysize(obj4))

println("\n--- D=4: build_melitz_implicit_bundle with PURE PRODUCTION DEFAULTS ---")
ctx4 = let
    p, eq, cf = data4.primitives, data4.equilibrium, data4.counterfactual
    D = p.D; j = p.target_country
    moment_layout = MelitzMomentLayout(D)
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)
    f_pivot_c, f_pivot_idx, f_pivot_other = melitz_build_f_pivot_parts(D, outer_layout.f_free_lin, A_pivot, c_full)
    sorted_tail_ctx = build_melitz_sorted_tail_context(data4.z_draws, p.sigma; theta_star=p.theta_star)
    (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
     w_prime=cf.w_prime, L=data4.L, expenditure=eq.expenditure, benchmark_cutoff=eq.cutoff,
     moment_layout=moment_layout, X_data=data4.equilibrium.trade_flow, c_full=c_full, A_pivot=A_pivot,
     jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
     f_pivot_c=f_pivot_c, f_pivot_idx=f_pivot_idx, f_pivot_other=f_pivot_other,
     outer_parameterization=:logf, technology_coordinate=:logA,
     inner_loop_opt=joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt"),
     outer_loop_opt=joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"),
     moment_backend=:sorted_tail_serial, sorted_tail_ctx=sorted_tail_ctx)
end
obj_implicit4 = build_melitz_implicit_bundle(ctx4, data4.z_draws, theta4;
    delta=1.0, find_smallest=true,
    inner_loop_opt=joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt"),
    outer_loop_opt=joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"),
    policy=CappedEvaluation(10.0))
println("concrete bundle type      = ", typeof(obj_implicit4))
println("has .H field?              = ", :H in fieldnames(typeof(obj_implicit4)))
println("has .G field?              = ", :G in fieldnames(typeof(obj_implicit4)))

println("\n--- counters after CONSTRUCTION ONLY ---")
println(melitz_backend_counters_snapshot())

# ---------------------------------------------------------------------------
# Real inner solve at D=4
# ---------------------------------------------------------------------------
println("\n--- D=4: real inner KNITRO solve (melitz_bundle_inner_loop: prepare-then-solve, the real production composition) ---")
melitz_backend_counters_reset!()
val, x, nStatus = melitz_bundle_inner_loop(obj4, theta4)
println("nStatus=", nStatus, "  val=", val, "  |x|=", length(x))
println("counters after ONE real D=4 inner solve:")
println(melitz_backend_counters_snapshot())

# ---------------------------------------------------------------------------
# REAL D=20/W=80,000 production fixture -- identical recipe to real campaign scripts
# (scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl /
# melitz_fixedqA_middleloop_experiment_2026-07-30.jl), including forbid_dense_fallback=true,
# which is what real campaigns actually pass.
# ---------------------------------------------------------------------------
println("\n" * "="^90)
println("--- REAL D=20/W=80,000 production fixture (real_data/noah_D20) ---")
REPO2 = dirname(@__DIR__)
OUTDIR2 = joinpath(REPO2, "docs", "key_results")

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR2, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0_d20 = theta_q_rows[("realD20_seed1_W80000", 0.5)]

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
println("focal country index = ", focal); flush(stdout)

melitz_backend_counters_reset!()
rss0 = rss_mb()
obj20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=CappedEvaluation(10.0), outer_parameterization=:logcutoff)
rss1 = rss_mb()
ctx_d20 = obj20.γ
D20 = ctx_d20.D
println("concrete bundle type (D=20,W=80000) = ", typeof(obj20))
println("has .H/.G/.K/.moments! field? = ", (:H in fieldnames(typeof(obj20)), :G in fieldnames(typeof(obj20)),
    :K in fieldnames(typeof(obj20)), Symbol("moments!") in fieldnames(typeof(obj20))))
println("RSS before/after construct  = ", round(rss0,digits=1), " / ", round(rss1,digits=1), " MB  (delta=", round(rss1-rss0,digits=2), ")")
println("Base.summarysize(obj20) bytes = ", Base.summarysize(obj20))
for fld in fieldnames(typeof(obj20))
    v = getfield(obj20, fld)
    if v isa AbstractArray
        @printf("  %-28s %-20s size=%-14s bytes=%d\n", fld, typeof(v), string(size(v)), Base.summarysize(v))
    end
end
W = 80_000; M = D20^2 + 1
println("\nImplied bytes of a legacy dense H=[K|1|G] at D=20,W=80000: W*(M+2)*8 = ", W*(M+2)*8, " bytes = ", round(W*(M+2)*8/1024^3, digits=3), " GiB")
println("construction counters (should be all-zero): ", melitz_backend_counters_snapshot())

println("\n--- D=20 real inner solve (real anchor theta) + counters ---")
melitz_backend_counters_reset!()
obj20.use_cached_x = false; obj20.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj20, theta0_d20)
@printf("D20 base-point solve: %.2fs  Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", time() - t0, lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
println("counters after ONE real D=20/W=80000 inner solve:")
println(melitz_backend_counters_snapshot())
x0_d20 = copy(lfd0.dual_x)

# ---------------------------------------------------------------------------
# Fixed-q A middle-loop shakedown: REAL production entry point
# (solve_melitz_fixed_q_A_profile_v2, fixed_q_a_middle_loop.jl), >=20 unique A evaluations,
# fixed at the anchor's own q (a trivially-feasible middle point) so no extra cliff-geometry
# machinery is needed -- still exercises the real production middle-loop driver end to end.
# ---------------------------------------------------------------------------
println("\n--- fixed-q A middle loop: REAL solve_melitz_fixed_q_A_profile_v2, >=20 evals ---")
session_d20 = MelitzInnerSession(obj20, ctx_d20, CappedEvaluation(10.0))
nA20 = D20^2 - 1
theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
A_free0_d20 = theta_plain0_d20[2:1+nA20]
_, _, _, _, q0_d20 = expand_free_theta_logcutoff(theta_plain0_d20, ctx_d20)
gpj0_d20 = exp(theta_plain0_d20[1])

melitz_backend_counters_reset!()
tmid = time()
result_mid = solve_melitz_fixed_q_A_profile_v2(session_d20, q0_d20, gpj0_d20, A_free0_d20, ctx_d20;
    coordinate=:logA, policy=CappedEvaluation(10.0), max_evals=25, box=0.1,
    outer_loop_opt=joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt"),
    theta_fixed_q_for_constraints=theta0_d20)
println("middle-loop wall = ", round(time()-tmid, digits=2), "s ; n_evals in log = ", length(result_mid.eval_log))
println("counters after the REAL profiled-A middle-loop run:")
mid_counters = melitz_backend_counters_snapshot()
println(mid_counters)
println("unique A points evaluated (fc-kind entries) = ", count(e -> e.kind == :fc, result_mid.eval_log))

println("\nDONE.")
