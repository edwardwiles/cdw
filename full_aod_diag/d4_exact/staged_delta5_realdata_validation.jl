# ============================================================================
# Real D=20/W=80,000 validation of the incumbent-seeding fix (task §3-4), the
# reusable-context refactor (task §5-6), AND the direction/gamma-bounds fix
# (addendum), run together against a genuine calibration start point.
#
# Start point mirrors the D=20 canonical-rerun frontier's own "Start A"
# (c_canon_run_one.jl, gravity-fullA-d20-canonical-rerun worktree) EXACTLY:
# g_start = gp0 (UNPERTURBED calibration gamma'_focal -- the Frechet benchmark
# itself, ctx.θ0_up[3+D]), zfree_start = pivot_reduce(log(theta0_up's own REAL
# A_od block), pe). find_smallest=true, i.e. the real "upper"/larger-kappa
# direction per the EVIDENCED convention in direction_bounds.jl (NOT the
# find_smallest=false this script used last session, before the addendum's
# direction audit corrected it).
#
# The pre-existing organic_pathology/d2_startA_canon_stage_complete_neval158.jls
# checkpoint (the literal real kappa~0.08 delta=2 boundary point) remains
# undeserializable on this branch (schema mismatch with the CM/threading
# integration branch's checkpoint schema bump, not yet rebased onto -- see
# handoff doc §1/§15/§6). This script's own g_start=gp0 IS the same real,
# genuine calibration value that checkpoint's own "Start A" provenance used, so
# this is a faithful real-data validation of the SAME starting basin, just
# re-derived fresh rather than deserialized from that specific stale file.
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Dates

lp(xs...) = (println(xs...); flush(stdout))

STAGE_BUDGET = parse(Float64, get(ENV, "STAGE_BUDGET", "45.0"))
CKPT_ROOT = joinpath(@__DIR__, "results_staged_delta5_realdata_validation")
rm(CKPT_ROOT; recursive = true, force = true)

lp("=== REAL-DATA VALIDATION: incumbent fix + context reuse + direction/gamma-bounds fix === ",
   Dates.now(), " STAGE_BUDGET=", STAGE_BUDGET)

FIND_SMALLEST = true   # the real upper/larger-kappa direction -- see direction_bounds.jl

lp("\n", "="^80, "\n=== Building probe context to derive the genuine calibration start point ===\n", "="^80)
t0 = time()
probe = build_fullA_context(W = 80000, δ = 2.0, find_smallest = FIND_SMALLEST, draw_design = :pseudorandom, draw_seed = 20260719)
lp("probe ctx build wall=", round(time() - t0, digits = 1), "s")
ctx = probe.ctx; pe = probe.pe
D = ctx.D
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
g_start = gp0   # UNPERTURBED -- matches c_canon_run_one.jl's real "Start A" exactly
gF = frechet_benchmark_gp(ctx)
lo, hi = direction_gamma_bounds(ctx, FIND_SMALLEST)
lp("genuine calibration start: gp0=", gp0, " (== Frechet benchmark g_F=", gF, ") length(zfree0)=", length(zfree0))
lp("direction box (find_smallest=", FIND_SMALLEST, "): [", lo, ", ", hi, "]  (g_start sits at the box's own upper edge, closed-interval feasible)")

lp("\n", "="^80, "\n=== ARM A: staged 2->3->4->5, reuse_context=true (", STAGE_BUDGET, "s/stage) ===\n", "="^80)
t0 = time()
staged_reuse = run_staged_delta5_continuation("valA", g_start, zfree0; find_smallest = FIND_SMALLEST,
    delta_stages = [2.0, 3.0, 4.0, 5.0], stage_maxtime_real = STAGE_BUDGET,
    ckpt_root = joinpath(CKPT_ROOT, "reuse_true"), reuse_context = true)
t_reuse_total = time() - t0
lp("\nARM A TOTAL wall=", round(t_reuse_total, digits = 1), "s (includes the ~65-83s one-time context build)")
for s in staged_reuse.stages
    lp("  stage ", s.stage, " (delta=", s.delta, "): kappa=", s.kappa, " wall=", round(s.wall, digits = 1),
       "s n_eval=", s.n_eval, " n_rejected=", s.n_rejected, " knitro_status=", s.knitro_status)
end

# ---- Monotonicity check: the actual invariant the incumbent-seeding fix guarantees ----
# best_gp, NOT kappa: kappa=1-gp^(sigma/(sigma-1)) is a strictly DECREASING function of
# gp for sigma>1, so under find_smallest=true (minimize gp, the real "upper"/larger-
# kappa direction), the tracked/protected quantity moving correctly is a NON-INCREASING
# best_gp (equivalently non-decreasing kappa) -- opposite arithmetic direction from last
# session's (find_smallest=false) run, same underlying invariant: the incumbent must
# never regress relative to its own genuinely-feasible start.
best_gps = [s.best_gp for s in staged_reuse.stages]
gp_monotonic = all(isnan(best_gps[i]) || isnan(best_gps[i-1]) || best_gps[i] <= best_gps[i-1] + 1e-9 for i in 2:length(best_gps))
kappas = [s.kappa for s in staged_reuse.stages]
kappa_monotonic = all(isnan(kappas[i]) || isnan(kappas[i-1]) || kappas[i] >= kappas[i-1] - 1e-9 for i in 2:length(kappas))
lp("\n", "="^80, "\n=== MONOTONICITY CHECK ===\n", "="^80)
lp("best_gp across stages (find_smallest=true -- non-increasing is correct): ", best_gps)
lp("kappa across stages (find_smallest=true -- non-decreasing is correct): ", kappas)
lp("best_gp monotonically non-increasing: ", gp_monotonic, "   kappa monotonically non-decreasing: ", kappa_monotonic)
lp((gp_monotonic && kappa_monotonic) ? "PASS: incumbent-seeding fix + direction fix both hold under a real staged upper run." :
                                        "FAIL: investigate further.")

# ---- ARM B: reuse_context=false (old per-stage-rebuild behavior), same budget ----
lp("\n", "="^80, "\n=== ARM B: staged 2->3->4->5, reuse_context=false (", STAGE_BUDGET, "s/stage) ===\n", "="^80)
t0 = time()
staged_norebuild = run_staged_delta5_continuation("valB", g_start, zfree0; find_smallest = FIND_SMALLEST,
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

lp("\nNumerical equivalence check (reuse should change wall time only): ",
   [s.kappa for s in staged_reuse.stages] == [s.kappa for s in staged_norebuild.stages] ? "MATCH" : "MISMATCH")

lp("\nDONE_STAGED_DELTA5_REALDATA_VALIDATION")
