# ============================================================================
# Continuation 10, Section 9 (finalize-architecture), Part C: exact COLD
# recheck of the SR1 canonical segment's terminal point
# (c10_canonical_benchmark.jl's "canon_sr1" run), in a GENUINELY SEPARATE
# Julia process (fresh ctx, fresh RNG seed, fresh inner-dual warm start reset
# to NaN -- i.e. warm=false, NOT reusing the checkpoint's saved dual state).
# This is a STRICTER check than the checkpoint/resume acceptance test's own
# "RESUME VALIDATION" (which deliberately reinjects the saved dual warm start
# to test checkpoint fidelity, warm=true) -- here we ask: does the inner CC
# dual solve, run completely from scratch with no warm start at all, land on
# the SAME Delta_dual/gravity/KKT/moment-residual at this exact (g,zfree)
# point? That's the genuine "is this point really a feasible/converged
# solution, not an artifact of one particular solve path" check Part C asks
# for.
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, LinearAlgebra

CKPT_SR1 = joinpath(D4X_ROOT, "results", "fullA_d4", "c10_finalize_canonical", "sr1")
ckpt_path = joinpath(CKPT_SR1, "canon_sr1_latest.jls")
println("Cold-recheck of: ", ckpt_path)
latest = load_checkpoint(ckpt_path)
println("checkpoint: reason=", latest.checkpoint_reason, " n_eval=", latest.n_eval,
        " knitro_iter=", latest.knitro_iter, " g=", latest.g,
        " verify_Delta_dual=", latest.verify_Delta_dual, " verify_gravity_value=", latest.verify_gravity_value,
        " verify_max_abs_moment_kkt_resid=", latest.verify_max_abs_moment_kkt_resid,
        " verify_moment_resid_norm=", latest.verify_moment_resid_norm)

# Fresh process, fresh ctx, fresh RNG (matching draw_seed for the SAME simulated economy),
# and a genuinely COLD inner solve (warm=false -> obj.x reset to NaN before KNITRO starts).
Random.seed!(latest.draw_seed)
ctx = d20_real_setup(W = latest.W, δ = latest.delta, find_smallest = latest.find_smallest)
pe = build_pivot_elimination(ctx)
x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf_final = x_free_from_w2(vcat(latest.g, latest.zfree))

t0 = time()
r_cold, meta_cold = evaluate_fullA_screened(xf_final, ctx; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = false, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_cold = time() - t0

d_delta = abs(r_cold.Delta_dual - latest.verify_Delta_dual)
d_grav = abs(r_cold.gravity_value - latest.verify_gravity_value)
d_kkt = abs(r_cold.max_abs_moment_kkt_resid - latest.verify_max_abs_moment_kkt_resid)
d_mr = abs(norm(r_cold.benchmark_unweighted_moment_mean) - latest.verify_moment_resid_norm)

@printf("[COLD RECHECK] wall=%.3fs inner_status=%d screen_status=%s\n", t_cold, r_cold.inner_status, string(meta_cold.screen_status))
@printf("[COLD RECHECK] recomputed: Delta_dual=%.15f gravity=%.15e kkt=%.15e |benchmark_unweighted_moment_mean|=%.15e\n",
        r_cold.Delta_dual, r_cold.gravity_value, r_cold.max_abs_moment_kkt_resid, norm(r_cold.benchmark_unweighted_moment_mean))
@printf("[COLD RECHECK] checkpoint:  Delta_dual=%.15f gravity=%.15e kkt=%.15e |benchmark_unweighted_moment_mean|=%.15e\n",
        latest.verify_Delta_dual, latest.verify_gravity_value, latest.verify_max_abs_moment_kkt_resid, latest.verify_moment_resid_norm)
@printf("[COLD RECHECK] |diff|: Delta_dual=%.3e gravity=%.3e kkt=%.3e moment_resid_norm=%.3e\n", d_delta, d_grav, d_kkt, d_mr)

println("DONE_COLDRECHECK")
