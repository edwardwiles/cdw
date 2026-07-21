# ============================================================================
# Bounded staged-delta5-continuation vs direct-baseline comparison (Phase E).
# Starts from the REAL delta=2 checkpoint's best_feasible point (same one
# used in Phase D's granular profiling), same total wall-clock budget for
# both arms. Budget kept modest (240s total per arm) given this session's
# overall time constraints -- a bounded, not exhaustive, real comparison.
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Serialization

lp(xs...) = (println(xs...); flush(stdout))

ckpt = deserialize(joinpath(@__DIR__, "organic_pathology", "d2_startA_canon_stage_complete_neval158.jls"))
b = ckpt.best_feasible
g0 = b.gp
zfree0 = b.w[2:end]
lp("Starting point: delta=2 checkpoint best_feasible, gp=", g0, " kappa=", b.kappa)

CKPT_ROOT = joinpath(@__DIR__, "results_staged_delta5")
rm(CKPT_ROOT; recursive = true, force = true)

STAGE_BUDGET = 60.0   # seconds per stage, 4 stages = 240s total
DIRECT_BUDGET = 240.0 # same total budget, single direct delta=5 attempt

lp("\n", "="^80, "\n=== ARM A: staged continuation (delta 2->3->4->5, ", STAGE_BUDGET, "s/stage) ===\n", "="^80)
t0 = time()
staged = run_staged_delta5_continuation("cmp", g0, zfree0; find_smallest = true,   # addendum
    # fix: find_smallest is now REQUIRED (no silent default). true = the real upper/
    # larger-kappa direction (direction_bounds.jl), matching this script's own "upper"
    # narrative and the source checkpoint's own provenance (d2_startA_canon, produced
    # under c_canon_run_one.jl's find_smallest=true). The pre-fix call here omitted the
    # argument entirely and (via run_staged_delta5_continuation's own then-hardcoded
    # `false`) silently ran the opposite, lower-kappa direction regardless.
    delta_stages = [2.0, 3.0, 4.0, 5.0], stage_maxtime_real = STAGE_BUDGET,
    ckpt_root = joinpath(CKPT_ROOT, "staged"))
t_staged_total = time() - t0
lp("\nSTAGED TOTAL wall=", round(t_staged_total, digits=1), "s  final kappa=", staged.final.kappa,
   " final knitro_status=", staged.final.knitro_status, " total n_eval=", sum(s.n_eval for s in staged.stages),
   " total n_rejected=", sum(s.n_rejected for s in staged.stages))
for s in staged.stages
    lp("  stage ", s.stage, " (delta=", s.delta, "): kappa=", s.kappa, " n_eval=", s.n_eval, " n_rejected=", s.n_rejected, " wall=", round(s.wall,digits=1), "s")
end

lp("\n", "="^80, "\n=== ARM B: direct baseline (delta=5 straight, ", DIRECT_BUDGET, "s) ===\n", "="^80)
CKPT_DIRECT = joinpath(CKPT_ROOT, "direct")
mkpath(CKPT_DIRECT)
t0 = time()
direct = run_polish_checkpointed("cmp_direct", true, g0, zfree0;   # addendum fix: true = upper, matches staged's own direction above (was hardcoded `false` -- the wrong, lower-kappa direction -- pre-fix)
    maxtime_real = DIRECT_BUDGET, hessopt_tag = "sr1", W_in = 80000, delta_in = 5.0,
    draw_seed_in = 20260719, ckpt_dir = CKPT_DIRECT, checkpoint_interval_s = 30.0)
t_direct_total = time() - t0
lp("\nDIRECT TOTAL wall=", round(t_direct_total, digits=1), "s  kappa=", direct.kappa,
   " knitro_status=", direct.knitro_status, " n_eval=", direct.n_eval, " n_rejected=", direct.n_rejected)

lp("\n", "="^80, "\n=== VERDICT ===\n", "="^80)
lp("Staged:  kappa=", staged.final.kappa, "  wall=", round(t_staged_total,digits=1), "s  n_rejected=", sum(s.n_rejected for s in staged.stages))
lp("Direct:  kappa=", direct.kappa, "  wall=", round(t_direct_total,digits=1), "s  n_rejected=", direct.n_rejected)
better = isnan(staged.final.kappa) ? "direct" : (isnan(direct.kappa) ? "staged" : (staged.final.kappa > direct.kappa ? "staged" : "direct"))
lp("Higher kappa (better) at matched ~", STAGE_BUDGET*4, "s budget: ", better)

lp("\nDONE_STAGED_DELTA5_COMPARISON")
