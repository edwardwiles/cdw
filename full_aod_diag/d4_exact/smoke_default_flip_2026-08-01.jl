# Smoke test, 2026-08-01: does the REAL production driver, called with ZERO gravity kwargs
# (relying purely on the just-flipped defaults), reproduce the Brazil-Korea-excluded sigma=3
# calibration point (theta*=7.489399587926583), and does it actually solve?
#
# Uses the production data default (real_data/noah_D20/, NOT the frozen campaign snapshot -- this
# is a genuine "real production run" smoke test, not a campaign-scoped one) and the real
# calibration w_a vector from the supplied start_manifest.json (confirmed checksummed-identical
# data between real_data/noah_D20/ and the frozen snapshot in
# BRAZIL_KOREA_GRAVITY_EXCLUSION_PROVENANCE_2026-07-31.json).
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_lookup_production.jl", "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "cm_meanzc_lookup_production.jl", "cm_originzc_lookup_production.jl",
          "lfix_buffer_reuse.jl", "bandwidth_cache_policy.jl", "fast_range_screen.jl", "country_resolve.jl",
          "c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl", "json_lite.jl", "campaign_cell_io.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, DelimitedFiles
lp(xs...) = (println(xs...); flush(stdout))

lp("="^90)
lp("SMOKE TEST: real production driver, ZERO gravity kwargs (pure defaults) -- ", Dates.now())
lp("="^90)

# ---- sanity: what does the default resolve to, cheaply, before any context build? ----
default_cells = default_gravity_exclude_cells_brazil_korea()
lp(">> default_gravity_exclude_cells_brazil_korea() = ", default_cells, "  (expect [(3, 14)])")

const LAYOUT = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
const W = 500_000
const DRAW_DESIGN = :sobol_randomized
const DRAW_SEED = 20260719
const DELTA = 0.01

manifest = json_load("/tmp/claude-181517/-bbkinghome-edav-gravity-robustness/27d07afe-498a-460c-9edf-ff93c9abe8c2/scratchpad/bk_zip/extracted/brazil_korea_gravity_exclusion_and_starting_point_bug_2026-08-01/key_results/start_manifest.json")
st = manifest["starts"][1]
st["label"] == "start1_calibration" || error("expected start1_calibration")
w_a = jf64(st["w_transformed_a"])
h = string(hash(w_a), base = 16)
h == st["checksum_w_hash"] || error("checksum mismatch -- refusing to run")
lp(">> start1_calibration loaded and checksum-verified: gp=", w_a[1], " |w|=", length(w_a))

ckdir = mktempdir(; prefix = "smoke_default_flip_")
lp(">> ckpt_dir = ", ckdir)

t0 = time()
result = nothing
errored = false
try
    w_start = copy(w_a)
    # NOTE: exclude_diagonal_gravity / gravity_exclude_cells / sigmaHat DELIBERATELY OMITTED --
    # this is the entire point of the smoke test: confirm the FLIPPED DEFAULTS alone (not an
    # explicit override) reproduce the Brazil-Korea-excluded sigma=3 calibration point.
    local result_inner = run_polish_checkpointed_unified("smoke_default_flip", true, w_start;
        layout = LAYOUT, theta_lo = NaN, theta_hi = NaN,
        maxtime_real = 25.0, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        draw_design_in = DRAW_DESIGN, ckpt_dir = ckdir, checkpoint_interval_s = 3600.0,
        resume_from = nothing, destination_sample = :exclude_row)
    global result = result_inner
catch e
    global errored = true
    lp(">> *** EXCEPTION *** ", typeof(e), ": ", sprint(showerror, e)[1:min(end, 2000)])
    for (i, fr) in enumerate(stacktrace(catch_backtrace()))
        i <= 15 && lp("     [", i, "] ", fr)
    end
end
lp(">> wall = ", round(time() - t0, digits = 1), "s  errored=", errored)
# NOTE (2026-08-01, post-run fix): originally `lp(">> result = ", result)` -- result is a
# NamedTuple that nests the full ctx (all its large matrices) and printing it flooded ~1.6GB of
# output, which got the background job SIGKILL'd on first run. Print only the scalar summary
# fields actually needed to confirm success.
if result !== nothing
    lp(">> result: knitro_status=", get(result, :knitro_status, get(result, :status, "?")),
       "  n_eval=", get(result, :n_eval, "?"))
end
lp("DONE")
