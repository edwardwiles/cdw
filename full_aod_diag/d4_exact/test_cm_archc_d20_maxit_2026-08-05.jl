# 2026-08-05: decisive test of the "iteration-budget, not infeasibility" hypothesis for the
# two-family nStatus=-400 result seen in test_cm_archc_d20_fixedstate_2026-08-05.jl at both
# W=100,000 and W=300,000 (IDENTICAL failure at both scales, ruling out finite-sample/W-sensitivity
# as the explanation -- if it were noise, the single-family case's own W=100k->W=300k improvement
# from -103 to 0 would have had an analogue here, and it didn't).
#
# knitro_status.jl's own table says nStatus=-400 IS `KN_RC_ITER_LIMIT_FEAS`, `is_feasible_result=
# true`: "Iteration limit (maxit) reached before full convergence; a feasible point WAS found." And
# full_aod_diag/ek_inner.opt (the inner-solve options file every archC_base_state call loads via
# `KN_load_param_file`) sets `maxit 100` -- a fixed cap tuned for the SINGLE-family (190-variable)
# dual problem. archC_base_state's own acceptance list is `(0,-100,-101,-103)` -- narrower than
# many other gates/benches in this repo that also accept -400/-401/-402 -- so a genuinely feasible
# but not-yet-fully-converged two-family (380-variable, 2x the dual system) point gets rejected as
# a "solve failure" even though it isn't one.
#
# This script re-runs the identical W=100,000 two-family fixed-state check but with `maxit` raised
# from 100 to 2000 (ek_inner_maxit2000_2026-08-05.opt, a copy of ek_inner.opt with only that one
# line changed) via d20_real_setup's own `inner_loop_opt` kwarg. If the two-family solve now
# reaches a fully-accepted status (0/-100/-101/-103) instead of -400, that confirms this was purely
# an iteration-budget artifact of doubling the moment count -- NOT a moment-specification bug and
# NOT genuine economic infeasibility.
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
const MAXIT_OPT = joinpath(D4X, "..", "ek_inner_maxit2000_2026-08-05.opt")
@assert isfile(MAXIT_OPT) "missing $MAXIT_OPT"

ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row,
                      inner_loop_opt = MAXIT_OPT)
x_free_calib = ctx.θ0_up[ctx.free_idx]
lp("Context built. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat, " inner_loop_opt=", ctx.obj.inner_loop_opt)

const L = 10
probs_ = collect(range(1 / L, (L - 1) / L, length = L))

for include_tm in (false, true)
    lp("="^100)
    lp("include_truncated_moment=$include_tm -- maxit=2000 (was 100)")
    lp("="^100)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs_,
        include_truncated_moment = include_tm,
        moment_representation = :operator, inner_fg_backend = :cm_lookup,
        use_archB_moments = false)
    lp("n_families=", pcx.cctx.n_families, " ncm=", pcx.cctx.ncm)
    t0 = time()
    try
        base = archC_base_state(copy(x_free_calib), pcx.ctx_cm, pcx.cctx)
        lp("RESULT include_tm=$include_tm: inner_status=", base.inner_status, " (", time() - t0, "s)")
        println(base.inner_status in (0, -100, -101, -102, -103) ? "PASS" : "FAIL", "  include_tm=$include_tm feasible at raw calibration point (inner_status=$(base.inner_status))")
    catch e
        lp("RESULT include_tm=$include_tm: EXCEPTION -- ", sprint(showerror, e))
        println("FAIL  include_tm=$include_tm threw an exception (see above)")
    end
end
