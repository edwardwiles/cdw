# ============================================================================
# Continuation 10, Section 9 (finalize-architecture), Part C: canonical
# benchmark through the ACTUAL driver (c10_d20_production_driver.jl), post
# Part A wiring (structured moment materialization + BLAS KKT/moment-residual
# swap). Real D=20/W=80,000 data. Reports:
#   1. one exact (hard-max) cold value evaluation
#   2. one warm inner solve at the same point
#   3. one full outer composite gradient evaluation
#   4. a real SR1 outer-iteration segment (target ~20-50 KNITRO iterations)
#   5. a real L-BFGS outer-iteration segment at the SAME starting point/budget
#      (Part B: SR1 vs L-BFGS side-by-side, time-to-best-feasible)
# The exact cold-recheck of the SR1 segment's terminal point is done
# SEPARATELY (c10_canonical_coldrecheck.jl), reusing run_profile_checkpointed's
# own resume-validation mechanism in a genuinely separate process, per this
# investigation's checkpoint/resume acceptance-test discipline.
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, LinearAlgebra, Dates

function rss_mb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                return parse(Int, split(line)[2]) / 1024
            end
        end
    catch
    end
    return NaN
end

lp(xs...) = (println(xs...); flush(stdout))

lp("=== CANONICAL BENCHMARK (Continuation 10 Section 9) === ", Dates.now())
lp("RSS at start: ", round(rss_mb(), digits=1), " MB")

Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = 80000, find_smallest = true, δ = 1.0)
lp("ctx build wall=", round(time()-t0, digits=1), "s  RSS=", round(rss_mb(), digits=1), " MB  screen_setup_wall=", ctx.screen_setup_wall)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0 * 1.01, zfree0)
xf0 = x_free_from_w2(w0)

# ---- 1. one exact (hard-max) cold value evaluation ----
t0 = time()
r_cold, meta_cold = evaluate_fullA_screened(xf0, ctx; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = false, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_cold_val = time() - t0
lp("[1. exact cold value eval] wall=", round(t_cold_val, digits=3), "s  Delta_dual=", r_cold.Delta_dual,
   "  inner_status=", r_cold.inner_status, "  screen_status=", meta_cold.screen_status, "  RSS=", round(rss_mb(),digits=1), " MB")

# ---- 2. one warm inner solve at the SAME point ----
t0 = time()
r_warm, meta_warm = evaluate_fullA_screened(xf0, ctx; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = true, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_warm_val = time() - t0
lp("[2. warm inner solve, same pt] wall=", round(t_warm_val, digits=3), "s  Delta_dual=", r_warm.Delta_dual, "  inner_status=", r_warm.inner_status)

# ---- 3. one full outer composite gradient evaluation ----
base = compressed_base_state(xf0, ctx)
t0 = time()
gfull, gmeta = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_grad = time() - t0
lp("[3. full outer gradient] wall=", round(t_grad, digits=3), "s  norm(gfull)=", norm(gfull), "  tie_fallback=", gmeta.tie_fallback, "  RSS=", round(rss_mb(),digits=1), " MB")

# ---- 4. real SR1 outer-iteration segment, target ~20-50 KNITRO iterations ----
CKPT_SR1 = joinpath(D4X_ROOT, "results", "fullA_d4", "c10_finalize_canonical", "sr1")
rm(CKPT_SR1; recursive = true, force = true); mkpath(CKPT_SR1)
lp("\n=== SR1 segment (canonical, target ~20-50 KNITRO outer iterations) ===")
t0 = time()
res_sr1 = run_profile_checkpointed("canon_sr1", gp0 * 1.01, false, zfree0;
    maxtime_real = 480.0, hessopt_tag = "sr1", W_in = 80000, delta_in = 1.0, draw_seed_in = 20260719,
    ckpt_dir = CKPT_SR1, checkpoint_interval_s = 45.0)
t_sr1 = time() - t0
lp("[SR1] wall_ext=", round(res_sr1.wall_ext, digits=1), "s  n_eval=", res_sr1.n_eval,
   "  n_grad_calls=", res_sr1.n_grad_calls, "  knitro_status=", res_sr1.knitro_status,
   "  knitro_iter=", res_sr1.final_checkpoint.knitro_iter, "  RSS=", round(rss_mb(),digits=1), " MB")
if res_sr1.best !== nothing
    lp("[SR1] best: Delta=", res_sr1.best.Delta_dual, " found_at_eval=", res_sr1.best.n_eval, " t_elapsed=", round(res_sr1.best.t_elapsed, digits=1), "s")
end
lp("[SR1] final ckpt path: ", res_sr1.ckpt_path)

# ---- 5. real L-BFGS outer-iteration segment, SAME starting point + budget ----
CKPT_LBFGS = joinpath(D4X_ROOT, "results", "fullA_d4", "c10_finalize_canonical", "lbfgs")
rm(CKPT_LBFGS; recursive = true, force = true); mkpath(CKPT_LBFGS)
lp("\n=== L-BFGS segment (Part B side-by-side, SAME starting point + budget) ===")
t0 = time()
res_lbfgs = run_profile_checkpointed("canon_lbfgs", gp0 * 1.01, false, zfree0;
    maxtime_real = 480.0, hessopt_tag = "lbfgs", W_in = 80000, delta_in = 1.0, draw_seed_in = 20260719,
    ckpt_dir = CKPT_LBFGS, checkpoint_interval_s = 45.0)
t_lbfgs = time() - t0
lp("[LBFGS] wall_ext=", round(res_lbfgs.wall_ext, digits=1), "s  n_eval=", res_lbfgs.n_eval,
   "  n_grad_calls=", res_lbfgs.n_grad_calls, "  knitro_status=", res_lbfgs.knitro_status,
   "  knitro_iter=", res_lbfgs.final_checkpoint.knitro_iter, "  RSS=", round(rss_mb(),digits=1), " MB")
if res_lbfgs.best !== nothing
    lp("[LBFGS] best: Delta=", res_lbfgs.best.Delta_dual, " found_at_eval=", res_lbfgs.best.n_eval, " t_elapsed=", round(res_lbfgs.best.t_elapsed, digits=1), "s")
end
lp("[LBFGS] final ckpt path: ", res_lbfgs.ckpt_path)

lp("\n=== SIDE-BY-SIDE VERDICT ===")
if res_sr1.best !== nothing && res_lbfgs.best !== nothing
    lp("SR1   time-to-best-feasible: ", round(res_sr1.best.t_elapsed, digits=1), "s -> Delta=", res_sr1.best.Delta_dual, " (", res_sr1.best.n_eval, " evals, ", res_sr1.final_checkpoint.knitro_iter, " outer iters over ", round(res_sr1.wall_ext,digits=1), "s total)")
    lp("LBFGS time-to-best-feasible: ", round(res_lbfgs.best.t_elapsed, digits=1), "s -> Delta=", res_lbfgs.best.Delta_dual, " (", res_lbfgs.best.n_eval, " evals, ", res_lbfgs.final_checkpoint.knitro_iter, " outer iters over ", round(res_lbfgs.wall_ext,digits=1), "s total)")
end

lp("\nDONE_CANONICAL_BENCHMARK")
