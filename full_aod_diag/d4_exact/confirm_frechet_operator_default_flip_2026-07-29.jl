# Confirmation (2026-07-29): after flipping build_cm_frechet_production_context's
# moment_representation default to :operator, confirm that the DEFAULT (no kwarg passed --
# exactly how the real production driver run_cm_upper_checkpointed calls this function) now
# actually builds an OperatorPsiBundle, and that a real inner solve through it still works and
# matches the (still-supported, explicit-opt-in) :dense_reference path at real D=20 scale.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "context_real_d20.jl",
          "lfix_factorized_workspace.jl", "lfix_factorized.jl", "cm_screen_bridge.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl"]
    include(joinpath(_D4E, f))
end
using Printf

lp(xs...) = (println(xs...); flush(stdout))
lp("Building real D=20 context (W=80000, delta=1.0, destination_sample=:exclude_row)...")
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]

lp("="^90)
lp("Structural check: build_cm_frechet_production_context with NO moment_representation kwarg")
lp("(exactly how run_cm_upper_checkpointed, the real production driver, calls it)")
lp("="^90)
pcx_default = build_cm_frechet_production_context(ctx, CS; L = 50, contrasts = :anchored,
    cm_hessian_backend = :structured)
obj_default = pcx_default.ctx_cm.obj
is_operator = obj_default isa OperatorPsiBundle
println("obj isa OperatorPsiBundle (default, no kwarg passed): ", is_operator)
is_operator || error("CONFIRMATION FAILED: omitting moment_representation did not yield an OperatorPsiBundle")

lp("="^90)
lp("Real inner solve through the DEFAULT-built context (no explicit moment_representation)")
lp("="^90)
level_targets = pcx_default.aug.level_targets
base_default = archC_frechet_base_state(x_free_calib, pcx_default.ctx_cm, pcx_default.cctx, level_targets)
@printf "  default-context solve: nStatus=%d  zeta*=%.10f\n" base_default.inner_status base_default.ζstar

lp("Cross-check against an explicit :dense_reference sibling built the OLD way...")
pcx_dense = build_cm_frechet_production_context(ctx, CS; L = 50, contrasts = :anchored,
    cm_hessian_backend = :structured, moment_representation = :dense_reference)
base_dense = archC_frechet_base_state(x_free_calib, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
@printf "  dense-reference solve: nStatus=%d  zeta*=%.10f\n" base_dense.inner_status base_dense.ζstar

zdiff = abs(base_default.ζstar - base_dense.ζstar)
ldiff = maximum(abs.(base_default.λstar .- base_dense.λstar))
@printf "  |Δζ*|=%.3e  max|Δλ*|=%.3e  status_match=%s\n" zdiff ldiff (base_default.inner_status == base_dense.inner_status)

ok = base_default.inner_status == base_dense.inner_status && zdiff < 1e-8
println(ok ? "CONFIRMATION PASSED: production default now builds+solves the no-H OperatorPsiBundle correctly" :
             "CONFIRMATION FAILED")
ok || exit(1)
