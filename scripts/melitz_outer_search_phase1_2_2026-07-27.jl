# Melitz outer-search session (governing prompt, 2026-07-27 continuation): Phases 1-2.
#
# Phase 1: one authoritative 20-thread timing table (kernels + complete operations) at real
# D=20/W=80,000, matched calibration/seed/dual-start/options/backend/JIT-warmup.
#
# Phase 2: exhaustive, nonoverlapping decomposition of one finite outer FC callback, using the
# `@melitz_profile`/`melitz_record_seconds_outcome!` instrumentation already built into this
# codebase's production hot path (src/melitz/profiling.jl), extended this session with four
# new categories closing named gaps (:fc_theta_expand, :fc_operator_merge,
# :fc_warm_start_resolve, :fc_cache_insert, plus per-KNITRO-callback-body timers
# :fc_inner_obj_eval/:fc_inner_dpsi_eval/:fc_inner_grad_eval/:fc_inner_hess_eval inside the
# MelitzCCBundle functor itself). Runs three matched variants at the IDENTICAL theta:
#   A. direct production fixed-point inner solve (matches the prior closure doc's own
#      Phase 4 matched benchmark methodology exactly: build_melitz_psi_bundle_from_calibration
#      + melitz_recover_lfd, capped options file, single thread).
#   C. complete outer FC (cb_F!) with candidate registration ENABLED (the real production
#      path) -- profiled with MELITZ_PROFILE[]=true, categories reset immediately before.
#   B (derived, not a separately re-run construction): fc_total minus fc_candidate_registration
#      from the SAME profiled C run -- exact by construction (register_live_candidate! is a
#      single, non-overlapping sub-step of cb_F!'s try body, confirmed by reading the source),
#      not an approximation from a second, differently-built callback set.
#
# Threading policy (session-level instruction): JULIA_NUM_THREADS=20, BLAS threads=1 for the
# 20-thread rows; a separate single-thread process/run supplies the serial baseline rows
# (comparing wall-clock BETWEEN Julia processes with different -t values is fine; comparing
# within one process after changing Threads.nthreads() is not possible in Julia).

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

mutable struct MockEvalRequestC
    x::Vector{Float64}
end
mutable struct MockEvalResultC
    obj::Vector{Float64}
    c::Vector{Float64}
    objGrad::Vector{Float64}
    jac::Vector{Float64}
end

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const NTHREADS = Threads.nthreads()
println("Threads.nthreads() = ", NTHREADS, "  BLAS threads = ", LinearAlgebra.BLAS.get_num_threads())

rows_timing = Vector{NamedTuple}()
rows_fc = Vector{NamedTuple}()

# ----------------------------------------------------------------------------------------
# Real D=20 fixture -- IDENTICAL construction/seed/options to
# docs/key_results/melitz_phase12_closure_benchmarks_2026-07-27.csv's own real-D20 rows.
# ----------------------------------------------------------------------------------------
real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = findfirst(==("fra"), countries)
observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
z_draws = pareto_draws(80_000, calib.D, calib.theta_star; seed=calib.seed)
p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_serial)
theta0_20 = melitz_reduce_theta(p20, ctx20)
n20 = length(theta0_20)
op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
obj20 = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
    outer_constr_index=ctx20.moment_layout.num_moments + 1,
    lower_limit=-10.0,   # explicit evaluation-cap wire-up -- build_melitz_cc_bundle no longer has a dangerous default (see cc_bundle.jl docstring); matches this session's standard delta_evaluation_cap=10.0 convention so a poorly-conditioned point fails fast instead of grinding on an uncapped inner solve
    inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt,
    hessian_backend=NTHREADS > 1 ? :structured_parallel : :structured_serial)
r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
x0_20 = r0_20.dual_x
println("D=20: Delta0=", r0_20.Delta, " nStatus=", r0_20.nStatus, " n_theta=", n20)

# ============================================================================================
# PHASE 1: kernels
# ============================================================================================
println("\n== Phase 1: kernel timings (n_threads=$NTHREADS) ==")

# -- sorted explicit moment construction (dense reference, for comparison only) --
# -- matrix-free operator update at theta0 (theta-expand + merge sweep combined) --
b_upd = @allocated melitz_update_operator_at_theta!(op20, theta0_20, ctx20)
t_upd = @elapsed melitz_update_operator_at_theta!(op20, theta0_20, ctx20)
push!(rows_timing, (kernel="operator_update_total", n_threads=NTHREADS, seconds=t_upd, bytes=b_upd, calls_per_fc=1))
println(@sprintf("operator update (theta-expand+merge): %.6fs, %d bytes", t_upd, b_upd))

# -- matrix-free G*x, G'*v --
zeta0, mu0 = x0_20[1], @view x0_20[2:end]
arg0buf = similar(obj20.arg0)
mul_G!(arg0buf, op20, zeta0, mu0)
t_Gx = @elapsed mul_G!(arg0buf, op20, zeta0, mu0)
b_Gx = @allocated mul_G!(arg0buf, op20, zeta0, mu0)
push!(rows_timing, (kernel="matrix_free_Gx", n_threads=NTHREADS, seconds=t_Gx, bytes=b_Gx, calls_per_fc=missing))
println(@sprintf("mul_G! (G*x): %.6fs, %d bytes", t_Gx, b_Gx))

gmu_buf = zeros(length(mu0))
mul_Gt!(gmu_buf, op20, obj20.arg1)
t_Gtv = @elapsed mul_Gt!(gmu_buf, op20, obj20.arg1)
b_Gtv = @allocated mul_Gt!(gmu_buf, op20, obj20.arg1)
push!(rows_timing, (kernel="matrix_free_Gtv", n_threads=NTHREADS, seconds=t_Gtv, bytes=b_Gtv, calls_per_fc=missing))
println(@sprintf("mul_Gt! (G'*v): %.6fs, %d bytes", t_Gtv, b_Gtv))

# -- structured Hessian + packed-Hessian callback (via the functor's own h= branch) --
gbuf = zeros(n20 == 0 ? 1 : length(x0_20))  # placeholder, unused shape-wise below
n_h = (length(x0_20) * (length(x0_20) + 1)) ÷ 2
hbuf = zeros(n_h)
obj20(x0_20, Float64[], Float64[]; h=hbuf)   # warmup
t_hess = @elapsed obj20(x0_20, Float64[], Float64[]; h=hbuf)
b_hess = @allocated obj20(x0_20, Float64[], Float64[]; h=hbuf)
push!(rows_timing, (kernel="structured_hessian_callback", n_threads=NTHREADS, seconds=t_hess, bytes=b_hess, calls_per_fc=missing))
println(@sprintf("structured Hessian callback (h= branch): %.6fs, %d bytes", t_hess, b_hess))

# -- complete outer gradient (serial always available; parallel iff NTHREADS>1) --
gfun20 = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
gbuf20 = zeros(n20)
gfun20(gbuf20, theta0_20, ctx20, obj20, x0_20)
t20 = @elapsed gfun20(gbuf20, theta0_20, ctx20, obj20, x0_20)
b20 = @allocated gfun20(gbuf20, theta0_20, ctx20, obj20, x0_20)
push!(rows_timing, (kernel="complete_outer_gradient_serial", n_threads=NTHREADS, seconds=t20, bytes=b20, calls_per_fc=1))
println(@sprintf("complete outer gradient (serial call, ambient %d threads): %.6fs, %d bytes", NTHREADS, t20, b20))

gbuf20_check = copy(gbuf20)
if NTHREADS > 1
    gfun20p = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
    gbuf20p = zeros(n20)
    gfun20p(gbuf20p, theta0_20, ctx20, obj20, x0_20)
    t20p = @elapsed gfun20p(gbuf20p, theta0_20, ctx20, obj20, x0_20)
    b20p = @allocated gfun20p(gbuf20p, theta0_20, ctx20, obj20, x0_20)
    push!(rows_timing, (kernel="complete_outer_gradient_parallel", n_threads=NTHREADS, seconds=t20p, bytes=b20p, calls_per_fc=1))
    println(@sprintf("complete outer gradient (parallel, %d threads): %.6fs, %d bytes  [speedup vs serial: %.2fx]", NTHREADS, t20p, b20p, t20 / t20p))
    println("serial/parallel agreement: max|diff| = ", maximum(abs.(gbuf20_check .- gbuf20p)))
end

# ============================================================================================
# PHASE 1: complete operations
# ============================================================================================
println("\n== Phase 1: complete operations ==")

# -- complete finite inner solve (direct production path, matches Phase 4 closure doc) --
inner_loop_opt_capped = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
obj20b, theta0_20b = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=calib.seed,
    inner_loop_opt=inner_loop_opt_capped, forbid_dense_fallback=true)
melitz_recover_lfd(obj20b, theta0_20b)   # warmup
t_inner = @elapsed lfd_res = melitz_recover_lfd(obj20b, theta0_20b)
push!(rows_timing, (kernel="complete_finite_inner_solve", n_threads=NTHREADS, seconds=t_inner, bytes=missing, calls_per_fc=1))
println(@sprintf("complete finite inner solve (direct, capped opts): %.6fs, nStatus=%d Delta=%.6e", t_inner, lfd_res.nStatus, lfd_res.Delta))

# -- complete finite FC + complete finite GA, profiled --
melitz_profile_reset!()
global MELITZ_PROFILE
MELITZ_PROFILE[] = true

m20 = 1   # :linear cutoff backend -> evalResult.c sized 1 (divergence row only)
delta_loose20 = max(r0_20.Delta * 5, 1e-3)
gbackend = NTHREADS > 1 ? :B_direct_argument_sorted_parallel : :B_direct_argument_sorted_serial
cbset_finite = melitz_build_finite_delta_callbacks(obj20, ctx20, delta_loose20, true;
    gradient_backend=gbackend, h=1e-4, cutoff_constraint_backend=:linear)
evalFinite = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
cbset_finite.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalFinite, nothing)   # warmup (JIT)
melitz_profile_reset!()
tfin = @elapsed cbset_finite.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalFinite, nothing)
rows_profile_fc = melitz_profile_summary()
println(@sprintf("\ncomplete finite FC (profiled, registration ENABLED = variant C): %.6fs, obj=%.6e", tfin, evalFinite.obj[1]))
melitz_profile_report(stdout; trajectory_total_s=tfin)
push!(rows_timing, (kernel="complete_finite_FC_variant_C", n_threads=NTHREADS, seconds=tfin, bytes=missing, calls_per_fc=1))

evalFiniteG = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
cbset_finite.cb_G!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalFiniteG, nothing)   # warmup
melitz_profile_reset!()
tga = @elapsed cbset_finite.cb_G!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalFiniteG, nothing)
println(@sprintf("\ncomplete finite GA immediately after FC at IDENTICAL theta (should hit FC-to-GA exact-cache): %.6fs", tga))
melitz_profile_report(stdout; trajectory_total_s=tga)
push!(rows_timing, (kernel="complete_finite_GA_after_FC_same_theta", n_threads=NTHREADS, seconds=tga, bytes=missing, calls_per_fc=1))

# -- AboveEvaluationCap FC --
delta_tight20 = r0_20.Delta / 100
cbset_cap = melitz_build_finite_delta_callbacks(obj20, ctx20, delta_tight20, true;
    gradient_backend=gbackend, h=1e-4, cutoff_constraint_backend=:linear)
evalCap = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
cbset_cap.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalCap, nothing)   # warmup
melitz_profile_reset!()
tcap = @elapsed cbset_cap.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalCap, nothing)
println(@sprintf("\nAboveEvaluationCap finite FC: %.6fs", tcap))
melitz_profile_report(stdout; trajectory_total_s=tcap)
push!(rows_timing, (kernel="above_eval_cap_FC", n_threads=NTHREADS, seconds=tcap, bytes=missing, calls_per_fc=1))

MELITZ_PROFILE[] = false

# ============================================================================================
# PHASE 2 output: exhaustive nonoverlapping FC decomposition CSV (variant C run above)
# ============================================================================================
fc_reg_row = filter(r -> r.category == :fc_candidate_registration, rows_profile_fc)
fc_reg_s = isempty(fc_reg_row) ? 0.0 : fc_reg_row[1].total_s
variant_B_s = tfin - fc_reg_s   # exact derivation, not a re-constructed second run -- see header
println(@sprintf("\nDerived variant B (fc_total - fc_candidate_registration, registration DISABLED equivalent): %.6fs", variant_B_s))
println(@sprintf("Variant A (direct production fixed-point inner solve): %.6fs", t_inner))
println(@sprintf("Variant C (complete FC, registration enabled): %.6fs", tfin))
println(@sprintf("Delta(C-B) = candidate registration cost = %.6fs (%.1f%% of C)", fc_reg_s, 100*fc_reg_s/tfin))
println(@sprintf("Delta(B-A) = screening+operator-update+warm-start overhead atop the bare inner solve = %.6fs (%.1f%% of C)", variant_B_s - t_inner, 100*(variant_B_s - t_inner)/tfin))

open(joinpath(OUTDIR, "melitz_phase2_fc_decomposition_2026-07-27.csv"), "w") do io
    println(io, "category,count,total_s,mean_ms,median_ms,p90_ms,max_ms,pct_of_fc_total")
    for r in rows_profile_fc
        println(io, join([r.category, r.count, r.total_s, r.mean_ms, r.median_ms, r.p90_ms, r.max_ms,
            round(100*r.total_s/tfin; digits=2)], ","))
    end
    println(io, "VARIANT_A_direct_inner_solve,1,$(t_inner),,,,,")
    println(io, "VARIANT_B_derived_fc_minus_registration,1,$(variant_B_s),,,,,")
    println(io, "VARIANT_C_complete_fc,1,$(tfin),,,,,100.0")
end
println("Wrote ", joinpath(OUTDIR, "melitz_phase2_fc_decomposition_2026-07-27.csv"))

open(joinpath(OUTDIR, "melitz_phase1_timing_table_2026-07-27.csv"), "w") do io
    cols = keys(rows_timing[1])
    println(io, join(cols, ","))
    for r in rows_timing
        println(io, join([r[c] for c in cols], ","))
    end
end
println("Wrote ", joinpath(OUTDIR, "melitz_phase1_timing_table_2026-07-27.csv"))
