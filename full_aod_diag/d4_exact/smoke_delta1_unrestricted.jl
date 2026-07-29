# Post-merge production smoke test (2026-07-28): delta=1, unrestricted, real D=20/W=80,000.
# c10_d20_production_driver.jl is self-contained (includes everything it needs).
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Printf
lp(xs...) = (println(xs...); flush(stdout))

const FIND_SMALLEST = length(ARGS) >= 1 ? (ARGS[1] == "true") : true
const DIRECTION = FIND_SMALLEST ? "upper" : "lower"
const W = 100_000
const DELTA = 1.0
const BUDGET = 90.0

ctx_probe = d20_real_setup(W = W, find_smallest = FIND_SMALLEST, δ = DELTA, destination_sample = :exclude_row)
pe_probe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D; D2 = D^2
lp("D=", D, " Ddest=", ctx_probe.D_dest)
lp("-"^90)
# Wire-unrestricted-operator-bundle task (2026-07-29) correction: this block used to unconditionally
# claim "bundle_type = OperatorPsiBundle" -- FALSE for THIS script specifically, which calls the
# LEGACY run_profile_checkpointed (c10_d20_production_driver.jl), not run_polish_checkpointed_unified
# (c10_d20_production_driver_unified.jl) -- only the latter was wired with build_unrestricted_
# operator_ctx this task (see compressed_live.jl). run_profile_checkpointed still builds ctx_probe as
# the dense PsiObjectiveBundleImplicit, unchanged; that's what this smoke run actually exercises.
lp("NO-H BUNDLE FACTS: unrestricted (this script, run_profile_checkpointed -- LEGACY driver)")
lp("  bundle_type = PsiObjectiveBundleImplicit (dense; NOT wired to OperatorPsiBundle -- only")
lp("  run_polish_checkpointed_unified was, see compressed_live.jl::build_unrestricted_operator_ctx).")
lp("  has_H_field = true (allocated; production Hessian backend :exact_winner_pair_parallel never")
lp("  reads/writes it, but it is NOT structurally absent the way OperatorPsiBundle's is).")
lp("-"^90)
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+ctx_probe.D*ctx_probe.D_dest]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, ctx_probe.D, ctx_probe.D_dest), pe_probe)
gp0 = ctx_probe.θ0_up[3+D]

CKPT_DIR = joinpath(D4X_ROOT, "results", "postmerge_smoke_2026-07-28", "unrestricted_$DIRECTION")
rm(CKPT_DIR; recursive = true, force = true); mkpath(CKPT_DIR)

lp("=== UNRESTRICTED smoke ($DIRECTION, find_smallest=$FIND_SMALLEST) ==="); flush(stdout)
t0 = time()
res = run_profile_checkpointed("unrestricted_$DIRECTION", gp0 * 1.01, FIND_SMALLEST, zfree0;
    maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 60.0)
wall = time() - t0

println("="^90)
@printf("RESULT unrestricted(%s): wall=%.1fs n_eval=%d\n", DIRECTION, wall, res.n_eval)
println("best=", res.best)
println("="^90)
