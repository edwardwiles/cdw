# ============================================================================
# Real D=20/W=80,000 validation of BOTH the incumbent-seeding fix (task §3-4)
# AND the reusable-context refactor (task §5-6), run together since they compose:
# a genuine staged 2->3->4->5 continuation from a real calibration-derived start,
# reporting (a) whether kappa is now monotonically non-decreasing across stages
# (the bug the task opened with: 0.0806->0.0281->0.0093->0.0040->0.0031, provably
# impossible for a correctly-tracked upper-bound incumbent) and (b) per-stage wall
# time with vs without context reuse.
#
# Start point: ctx.θ0_up's own genuine calibration A_od block (NOT the
# gravity-elimination pivot's z=0 reference point -- see memory note
# feedback-gravity-elimination-zero-is-not-calibration.md), gp0*1.01, the SAME
# recipe c10_canonical_benchmark.jl already uses and independently validated.
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Dates

lp(xs...) = (println(xs...); flush(stdout))

STAGE_BUDGET = parse(Float64, get(ENV, "STAGE_BUDGET", "45.0"))
CKPT_ROOT = joinpath(@__DIR__, "results_staged_delta5_realdata_validation")
rm(CKPT_ROOT; recursive = true, force = true)

lp("=== REAL-DATA VALIDATION: incumbent fix + context reuse === ", Dates.now(), " STAGE_BUDGET=", STAGE_BUDGET)

# ---- ARM A: reuse_context=true (the fix) -----------------------------------
lp("\n", "="^80, "\n=== Building probe context to derive a genuine calibration start point ===\n", "="^80)
t0 = time()
probe = build_fullA_context(W = 80000, δ = 2.0, find_smallest = false, draw_design = :pseudorandom, draw_seed = 20260719)
lp("probe ctx build wall=", round(time() - t0, digits = 1), "s")
ctx = probe.ctx; pe = probe.pe
D = ctx.D
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
g_start = gp0 * 1.01
lp("genuine calibration start: gp0=", gp0, " g_start=", g_start, " length(zfree0)=", length(zfree0))

lp("\n", "="^80, "\n=== ARM A: staged 2->3->4->5, reuse_context=true (", STAGE_BUDGET, "s/stage) ===\n", "="^80)
t0 = time()
staged_reuse = run_staged_delta5_continuation("valA", g_start, zfree0;
    delta_stages = [2.0, 3.0, 4.0, 5.0], stage_maxtime_real = STAGE_BUDGET,
    ckpt_root = joinpath(CKPT_ROOT, "reuse_true"), reuse_context = true)
t_reuse_total = time() - t0
lp("\nARM A TOTAL wall=", round(t_reuse_total, digits = 1), "s (includes the ~65-83s one-time context build)")
for s in staged_reuse.stages
    lp("  stage ", s.stage, " (delta=", s.delta, "): kappa=", s.kappa, " wall=", round(s.wall, digits = 1),
       "s n_eval=", s.n_eval, " n_rejected=", s.n_rejected, " knitro_status=", s.knitro_status)
end

# ---- Monotonicity check: the actual bug this whole exercise is about --------
kappas = [s.kappa for s in staged_reuse.stages]
monotonic = all(isnan(kappas[i]) || isnan(kappas[i-1]) || kappas[i] >= kappas[i-1] - 1e-9 for i in 2:length(kappas))
lp("\n", "="^80, "\n=== MONOTONICITY CHECK (task §3's core question) ===\n", "="^80)
lp("kappas across stages: ", kappas)
lp("Monotonically non-decreasing (required for a correctly-tracked upper bound): ", monotonic)
lp(monotonic ? "PASS: incumbent-seeding fix holds under a real staged run." :
               "FAIL: kappa regressed at some stage -- investigate further, the fix did not fully resolve this.")

# ---- ARM B: reuse_context=false (old per-stage-rebuild behavior), same budget ----
lp("\n", "="^80, "\n=== ARM B: staged 2->3->4->5, reuse_context=false (", STAGE_BUDGET, "s/stage) ===\n", "="^80)
t0 = time()
staged_norebuild = run_staged_delta5_continuation("valB", g_start, zfree0;
    delta_stages = [2.0, 3.0, 4.0, 5.0], stage_maxtime_real = STAGE_BUDGET,
    ckpt_root = joinpath(CKPT_ROOT, "reuse_false"), reuse_context = false)
t_norebuild_total = time() - t0
lp("\nARM B TOTAL wall=", round(t_norebuild_total, digits = 1), "s")
for s in staged_norebuild.stages
    lp("  stage ", s.stage, " (delta=", s.delta, "): kappa=", s.kappa, " wall=", round(s.wall, digits = 1),
       "s n_eval=", s.n_eval, " n_rejected=", s.n_rejected, " knitro_status=", s.knitro_status)
end

lp("\n", "="^80, "\n=== CONTEXT-REUSE WALL-TIME SAVINGS (task §6) ===\n", "="^80)
lp("Arm A (reuse_context=true)  total wall = ", round(t_reuse_total, digits = 1), "s")
lp("Arm B (reuse_context=false) total wall = ", round(t_norebuild_total, digits = 1), "s")
lp("Savings = ", round(t_norebuild_total - t_reuse_total, digits = 1), "s (",
   round(100 * (t_norebuild_total - t_reuse_total) / t_norebuild_total, digits = 1), "%)")
lp("Per-stage wall, arm A: ", [round(s.wall, digits = 1) for s in staged_reuse.stages])
lp("Per-stage wall, arm B: ", [round(s.wall, digits = 1) for s in staged_norebuild.stages])

lp("\nDONE_STAGED_DELTA5_REALDATA_VALIDATION")
