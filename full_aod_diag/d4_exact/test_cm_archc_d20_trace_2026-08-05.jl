# 2026-08-05: iteration-by-iteration KNITRO trace (outlev=3, maxit=60) for the two-family D20
# fixed-state case ONLY, to see whether the nStatus=-400 result (confirmed to persist even at
# maxit=2000, ~19 minutes wall-clock, refuting a simple "just needs a few more iterations"
# explanation) reflects slow-but-steady convergence toward feasibility/optimality, or a stalled/
# oscillating iterate -- the autodiff ground-truth check (test_cm_autodiff_groundtruth_2026-08-05.jl)
# already confirmed Architecture C's gradient/Hessian formulas are exact to machine precision, so
# this is purely about the SOLVE's own numerical behavior at this specific point, not a formula bug.
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random
lp(xs...) = (println(xs...); flush(stdout))

const W = 100_000
const TRACE_OPT = joinpath(D4X, "..", "ek_inner_trace_2026-08-05.opt")
@assert isfile(TRACE_OPT) "missing $TRACE_OPT"

ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row,
                      inner_loop_opt = TRACE_OPT)
x_free_calib = ctx.θ0_up[ctx.free_idx]
lp("Context built. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat)

const L = 10
probs_ = collect(range(1 / L, (L - 1) / L, length = L))

lp("="^100)
lp("include_truncated_moment=true -- outlev=3 trace, maxit=60")
lp("="^100)
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs_,
    include_truncated_moment = true,
    moment_representation = :operator, inner_fg_backend = :cm_lookup,
    use_archB_moments = false)
lp("n_families=", pcx.cctx.n_families, " ncm=", pcx.cctx.ncm)
try
    base = archC_base_state(copy(x_free_calib), pcx.ctx_cm, pcx.cctx)
    lp("RESULT inner_status=", base.inner_status)
catch e
    lp("RESULT: EXCEPTION -- ", sprint(showerror, e))
end
