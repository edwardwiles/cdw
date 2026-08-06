# 2026-08-05: fast, FIXED-STATE (not full outer optimization) check at real D20 data, comparing
# feasibility for BOTH family counts at the SAME raw calibration point, via the REAL no-dense-H
# production context (build_cm_production_context + archC_base_state, moment_representation=
# :operator, inner_fg_backend=:cm_lookup) -- isolates "is this genuinely infeasible under the
# corrected z^(sigma-1) exponent" from "does the full outer KNITRO optimization succeed from this
# starting point", which are different questions (the driver test found the latter fails at
# nStatus=-300; this checks whether that is REALLY a fresh feasibility characteristic of the
# corrected exponent, not an Architecture C defect -- D4 gates already rule out the latter at
# machine precision).
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

const W = 300_000   # 2026-08-05: bumped again from 100,000 -- raw-moment diagnostic (see
# test_cm_raw_moment_check_2026-08-05.jl) confirms both CM families average to ~0 under the raw
# measure at the true calibration point, with the deviation shrinking correctly as 1/sqrt(W)
# (0.0182 at W=20k -> 0.0059 at W=100k, vs sqrt(5)=2.24 theoretical ratio) -- strong evidence the
# moment specification+calibration are correct, not buggy. Single-family already PASSES at
# W=100,000 (inner_status=-103); two-family still threw nStatus=-400 there. Since two-family
# roughly doubles the moment count (190->380 at L=10), it plausibly needs a larger W than
# single-family's known ~80-100k threshold for the same feasibility margin -- testing directly
# rather than assuming, per this repo's standing rule against unverified W-sensitivity claims.
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
lp("Context built. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat)

const L = 10
probs_ = collect(range(1 / L, (L - 1) / L, length = L))

for include_tm in (false, true)
    lp("="^100)
    lp("include_truncated_moment=$include_tm -- real no-dense-H production context (build_cm_production_context, :operator/:cm_lookup)")
    lp("="^100)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs_,
        include_truncated_moment = include_tm,
        moment_representation = :operator, inner_fg_backend = :cm_lookup,
        use_archB_moments = false)
    lp("n_families=", pcx.cctx.n_families, " ncm=", pcx.cctx.ncm)
    t0 = time()
    try
        base = archC_base_state(copy(x_free_calib), pcx.ctx_cm, pcx.cctx)   # fresh copy each
        # iteration -- do NOT reuse/alias the same vector object across repeated archC_base_state
        # calls in one process (suspected in-place mutation of the caller's input by some
        # downstream KNITRO/obj.x warm-start path -- confirmed live: the SECOND iteration's own
        # exception message echoed astronomically large values completely unlike the real
        # calibration point, consistent with the first call's converged/diverged internal KNITRO
        # state overwriting the shared array via aliasing, not a fresh evaluation of the intended
        # point).
        lp("RESULT include_tm=$include_tm: inner_status=", base.inner_status, " (", time() - t0, "s)")
        println(base.inner_status in (0, -100, -101, -102, -103) ? "PASS" : "FAIL", "  include_tm=$include_tm feasible at raw calibration point (inner_status=$(base.inner_status))")
    catch e
        lp("RESULT include_tm=$include_tm: EXCEPTION -- ", sprint(showerror, e))
        println("FAIL  include_tm=$include_tm threw an exception (see above)")
    end
end
