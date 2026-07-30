# Gate 3 (2026-07-29 reduced-q validation session): make the D20 reduced direction and scalar
# derivative usable. One verified real-D20 state (noah_D20, focal=fra, seed=1, target=0.5,
# W=80,000 -- the SAME fixture Phase 13 of `docs/melitz_reduced_q_subspace_search_2026-07-29.md`
# used). NO D20 outer search is run here (governing prompt Rule: "Do not run a D20 outer search
# until all work in this gate is complete").

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0 = theta_q_rows[("realD20_seed1_W80000", 0.5)]

function load_realD20_calib()
    real_dir = joinpath(REPO, "real_data", "noah_D20")
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

function build_bundle()
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end

obj = build_bundle()
ctx = obj.γ
D = ctx.D; nA = D^2-1; nq = D^2-2
println("D=", D, "  nA=", nA, "  nq=", nq); flush(stdout)

obj.use_cached_x = false; obj.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj, theta0)
@printf("base-point solve: %.2fs  Delta0=%.6f  lfd_ok=%s\n", time()-t0, lfd0.Delta, lfd0.lfd_ok)
@assert lfd0.lfd_ok
x0 = copy(lfd0.dual_x)
flush(stdout)

# ================================================================================================
# Gate 3A: thread D20 direction construction
# ================================================================================================
println("\n" * "="^100); println("Gate 3A: threaded coordinatewise sweep vs serial reference"); println("="^100); flush(stdout)

bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)

GC.gc()
mem0 = Base.gc_live_bytes()
t_serial0 = time()
d_serial, g_q_serial = melitz_reduced_q_propose_direction(theta0, x0, ctx, obj; bandwidth_policy=bwpolicy)
t_serial = time() - t_serial0
mem_serial = Base.gc_live_bytes() - mem0
@printf("Serial reference: %.2fs, live-bytes delta=%d\n", t_serial, mem_serial); flush(stdout)

t_pool0 = time()
bundles = melitz_build_thread_bundle_pool(build_bundle, Threads.maxthreadid())
t_pool_build = time() - t_pool0
@printf("Bundle pool build (%d bundles): %.2fs\n", length(bundles), t_pool_build); flush(stdout)

GC.gc()
mem1 = Base.gc_live_bytes()
t_par0 = time()
d_par, g_q_par = melitz_reduced_q_propose_direction_threaded(theta0, x0, ctx, bundles; bandwidth_policy=bwpolicy)
t_par = time() - t_par0
mem_par = Base.gc_live_bytes() - mem1
@printf("Threaded (%d threads): %.2fs, live-bytes delta=%d, speedup=%.2fx\n", Threads.nthreads(), t_par, mem_par, t_serial/t_par); flush(stdout)

# Warm-cache repeated call (bundles already built, JIT already warm).
t_par2_0 = time()
d_par2, g_q_par2 = melitz_reduced_q_propose_direction_threaded(theta0, x0, ctx, bundles; bandwidth_policy=bwpolicy)
t_par2 = time() - t_par2_0
@printf("Threaded warm-cache repeat: %.2fs, speedup vs serial=%.2fx\n", t_par2, t_serial/t_par2); flush(stdout)

gq_diff = maximum(abs.(g_q_serial .- g_q_par))
d_diff = (d_serial === nothing || d_par === nothing) ? NaN : maximum(abs.(d_serial .- d_par))
@printf("Numerical agreement: max|g_q_serial - g_q_par| = %.3e   max|d_serial - d_par| = %.3e\n", gq_diff, d_diff)
flush(stdout)

gate3a_rows = [(metric="serial_wall_s", value=t_serial), (metric="bundle_pool_build_s", value=t_pool_build),
               (metric="threaded_wall_s", value=t_par), (metric="threaded_warm_repeat_s", value=t_par2),
               (metric="speedup_cold", value=t_serial/t_par), (metric="speedup_warm", value=t_serial/t_par2),
               (metric="live_bytes_delta_serial", value=Float64(mem_serial)),
               (metric="live_bytes_delta_threaded", value=Float64(mem_par)),
               (metric="max_abs_diff_g_q", value=gq_diff), (metric="max_abs_diff_d", value=isnan(d_diff) ? -1.0 : d_diff),
               (metric="n_threads", value=Float64(Threads.nthreads())), (metric="nq", value=Float64(nq))]
melitz_write_typed_counter_csv(joinpath(OUTDIR, "melitz_gate3a_d20_threading_benchmark_2026-07-29.csv"),
    ["metric", "value"], gate3a_rows)

d_use = d_par !== nothing ? d_par : d_serial
@assert d_use !== nothing "Gate 3A: no useful direction found at this D20 anchor -- cannot proceed to 3B/3C"

# ================================================================================================
# Gate 3B: three separate scales -- basis, derivative bandwidth, stage radius
# ================================================================================================
println("\n" * "="^100); println("Gate 3B: basis normalization (already target_switches=100 via existing infra)"); println("="^100)
sorted_ctx = ctx.sorted_tail_ctx
# Reuses melitz_build_reduced_q_stage directly (Rule 10) -- this gives both the r_basis-scaled
# b_q AND the REAL affine-feasible [s_lo,s_hi] interval (Phase 2's exact closed-form LP), not a
# hand-reconstructed approximation of it.
gate3_stage = melitz_build_reduced_q_stage(theta0, x0, ctx, obj, 1; bandwidth_policy=bwpolicy, target_switches=100)
@assert gate3_stage !== nothing "melitz_build_reduced_q_stage found no useful direction at this anchor"
b_q = gate3_stage.q_basis_free
r_basis = norm(b_q)
tp_basis, tm_basis = melitz_q_direction_two_sided_crossings(theta0, b_q, 1.0, ctx, sorted_ctx)
@printf("Basis: r_basis=%.6f  |b_q|=%.6f  crossings at s=+-1: (+%d,-%d)  affine-feasible s in [%.4f, %.4f]\n",
        r_basis, norm(b_q), tp_basis, tm_basis, gate3_stage.s_lo, gate3_stage.s_hi)
flush(stdout)

println("\n" * "="^100); println("Gate 3B: scalar derivative bandwidth h_grad candidate sweep"); println("="^100); flush(stdout)
h_candidates = [1.0, 0.5, 0.25, 0.125, 1/16, 1/32, 1/64, 1/128, 1/256]
bandwidth_rows = NamedTuple[]
best_h_grad = nothing
for h in h_candidates
    tp_h, tm_h = melitz_q_direction_two_sided_crossings(theta0, b_q, h, ctx, sorted_ctx)
    theta_p = copy(theta0); theta_p[1+nA+1:end] .+= h .* b_q
    theta_m = copy(theta0); theta_m[1+nA+1:end] .-= h .* b_q
    obj.use_cached_x = false; obj.x .= NaN
    lfd_p = melitz_recover_lfd(obj, theta_p)
    obj.use_cached_x = false; obj.x .= NaN
    lfd_m = melitz_recover_lfd(obj, theta_m)
    class_p = lfd_p.lfd_ok ? "FiniteSolved" : "NumericalFailure_or_unverified"
    class_m = lfd_m.lfd_ok ? "FiniteSolved" : "NumericalFailure_or_unverified"
    secant_fd, ok_p_fd, ok_m_fd = melitz_q_direct_block_secant(theta0, b_q, h, obj, ctx, x0; mode=:fixed_dual)
    secant_reopt = (lfd_p.lfd_ok && lfd_m.lfd_ok) ? (lfd_p.Delta - lfd_m.Delta) / (2h) : NaN
    relerr = (isfinite(secant_fd) && isfinite(secant_reopt) && secant_reopt != 0) ?
        abs(secant_fd - secant_reopt) / abs(secant_reopt) : NaN
    sign_agree = isfinite(secant_fd) && isfinite(secant_reopt) && sign(secant_fd) == sign(secant_reopt)
    both_finite = lfd_p.lfd_ok && lfd_m.lfd_ok
    both_switch = tp_h > 0 && tm_h > 0
    meets_criteria = both_finite && both_switch && sign_agree && isfinite(relerr) && relerr <= 0.10
    @printf("  h=%-10.6f plus_switch=%-4d minus_switch=%-4d class_p=%-14s class_m=%-14s fd=%-12.4e reopt=%-12.4e relerr=%-8.4f sign_agree=%-5s meets=%-5s\n",
            h, tp_h, tm_h, class_p, class_m, secant_fd, secant_reopt, relerr, sign_agree, meets_criteria)
    push!(bandwidth_rows, (h=h, plus_switches=tp_h, minus_switches=tm_h, total_switches=tp_h+tm_h,
          class_plus=class_p, class_minus=class_m, plus_affine_feasible=true, minus_affine_feasible=true,
          fixed_dual_secant=secant_fd, reoptimized_secant=secant_reopt, relative_error=relerr,
          sign_agreement=sign_agree, meets_criteria=meets_criteria))
    if meets_criteria && (best_h_grad === nothing || h > best_h_grad)
        global best_h_grad = h
    end
    flush(stdout)
end
melitz_write_typed_counter_csv(joinpath(OUTDIR, "melitz_gate3b_d20_bandwidth_calibration_2026-07-29.csv"),
    ["h", "plus_switches", "minus_switches", "total_switches", "class_plus", "class_minus",
     "plus_affine_feasible", "minus_affine_feasible", "fixed_dual_secant", "reoptimized_secant",
     "relative_error", "sign_agreement", "meets_criteria"], bandwidth_rows)

if best_h_grad === nothing
    println("\n*** NO h_grad CANDIDATE SATISFIES ALL ACCEPTANCE CRITERIA -- D20 extensive-margin derivative NOT validated. STOPPING before stage-radius calibration and Gate 4. ***")
    flush(stdout)
else
    @printf("\nSelected h_grad = %.6f (largest candidate meeting all criteria)\n", best_h_grad)
    flush(stdout)

    println("\n" * "="^100); println("Gate 3B: stage radius s_max calibration (one-sided sweep)"); println("="^100); flush(stdout)
    # scalar derivative at the SELECTED h_grad, for prediction
    secant_h, _, _ = melitz_q_direct_block_secant(theta0, b_q, best_h_grad, obj, ctx, x0; mode=:fixed_dual)
    s_candidates = [1/8, 1/4, 1/2, 1.0]
    radius_rows = NamedTuple[]
    best_s_max = nothing
    for s in s_candidates
        s_eff = min(s, gate3_stage.s_hi)   # intersect with the REAL affine-feasible interval
        theta_disp = copy(theta0); theta_disp[1+nA+1:end] .+= s_eff .* b_q
        tp_s, tm_s = melitz_q_direction_two_sided_crossings(theta0, b_q, s_eff, ctx, sorted_ctx)
        pred_dDelta = secant_h * s_eff   # linear prediction using the b_q-scaled derivative
        direct_onesided = melitz_q_direct_block_secant_one_sided(theta0, b_q, s_eff, obj, ctx, x0; sign=1)
        obj.use_cached_x = false; obj.x .= NaN
        lfd_disp = melitz_recover_lfd(obj, theta_disp)
        actual_dDelta = lfd_disp.lfd_ok ? lfd_disp.Delta - lfd0.Delta : NaN
        classification = lfd_disp.lfd_ok ? "FiniteSolved" : "NumericalFailure_or_unverified"
        relerr = (isfinite(actual_dDelta) && actual_dDelta != 0) ? abs(pred_dDelta - actual_dDelta) / abs(actual_dDelta) : NaN
        sign_agree = isfinite(actual_dDelta) && sign(pred_dDelta) == sign(actual_dDelta)
        meets = lfd_disp.lfd_ok && sign_agree && isfinite(relerr) && relerr <= 0.50
        @printf("  s=%-8.4f switches(+%d,-%d) pred_dDelta=%-12.4e actual_dDelta=%-12.4e class=%-14s relerr=%-8.4f sign_agree=%-5s meets=%-5s\n",
                s_eff, tp_s, tm_s, pred_dDelta, actual_dDelta, classification, relerr, sign_agree, meets)
        push!(radius_rows, (s=s_eff, plus_switches=tp_s, minus_switches=tm_s, direct_onesided_fixed_dual=direct_onesided,
              predicted_dDelta=pred_dDelta, reoptimized_Delta=lfd_disp.Delta, actual_dDelta=actual_dDelta,
              classification=classification, relative_error=relerr, sign_agreement=sign_agree, meets_criteria=meets))
        if meets && (best_s_max === nothing || s_eff > best_s_max)
            global best_s_max = s_eff
        end
        flush(stdout)
    end
    melitz_write_typed_counter_csv(joinpath(OUTDIR, "melitz_gate3b_d20_stage_radius_calibration_2026-07-29.csv"),
        ["s", "plus_switches", "minus_switches", "direct_onesided_fixed_dual", "predicted_dDelta",
         "reoptimized_Delta", "actual_dDelta", "classification", "relative_error", "sign_agreement", "meets_criteria"],
        radius_rows)

    if best_s_max === nothing
        println("\n*** NO s CANDIDATE SATISFIES THE ONE-SIDED STAGE-RADIUS CRITERIA -- no verified local trust interval. ***")
    else
        @printf("\nSelected one-sided stage radius s_max = %.6f\n", best_s_max)
    end
    flush(stdout)

    println("\n" * "="^100); println("Gate 3C: participation-switch-inclusive validation summary"); println("="^100)
    row = bandwidth_rows[findfirst(r -> r.h == best_h_grad, bandwidth_rows)]
    both_finite_and_switching = row.plus_switches > 0 && row.minus_switches > 0 && row.class_plus == "FiniteSolved" && row.class_minus == "FiniteSolved"
    @printf("At h_grad=%.6f: plus_switches=%d minus_switches=%d both_FiniteSolved=%s -- Gate 3C %s\n",
            best_h_grad, row.plus_switches, row.minus_switches, both_finite_and_switching, both_finite_and_switching ? "PASSES" : "FAILS")
    flush(stdout)
end

println("\nGate 3 complete.")
flush(stdout)
