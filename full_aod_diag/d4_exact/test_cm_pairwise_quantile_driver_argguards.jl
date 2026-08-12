# ================================================================================================
# ARGUMENT-TIME guards for the CM + pairwise-quantile campaign driver (family #7, 2026-08-12).
#
# Split out of `test_cm_pairwise_quantile_campaign_smoke.jl` on purpose: every check here fires
# BEFORE the driver builds a real-data D=20 context, so this file runs in seconds and can be
# re-run after any driver edit without paying ~80s per case for a context that is never used.
# The smoke file keeps the checks that genuinely need a live run (checkpoint round-trip, resume,
# resume-mismatch guards).
#
# THE POINT OF THE `hessopt` CHECK BEING HERE. The inner registration already refuses "Hessian
# builder supplied + option file not hessopt=exact", but only from INSIDE the outer KNITRO
# callback, where a thrown Julia error can surface as an opaque KN_RC_CALLBACK_ERR rather than the
# message (memory `feedback-archC-verified-state-direct-call-knitro-callback-err`). The driver now
# validates it at step 2 with KNITRO's OWN parser on a throwaway context. This file is what
# confirms that early path actually fires, and fires with a message that says what is wrong.
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_cm_pairwise_quantile_driver_argguards.jl
# ================================================================================================

const D4X = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "incumbent_logic.jl", "cm_checkpoint.jl", "cm_originzc_target_layout.jl",
          "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl",
          "country_resolve.jl", "cross_delta_cache.jl", "compressed_moments.jl",
          "canonical_price_precompute_workspace.jl", "hard_score_b_cache.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "lfix_buffer_reuse.jl",
          "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl", "bandwidth_cache_policy.jl",
          "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl", "dual_bank_ab_harness.jl",
          "reusable_context.jl", "organic_failure_capture.jl", "knitro_status.jl",
          "knitro_version_check.jl", "multistart_seed_generator.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl", "cm_pairwise_quantile_cplus.jl",
          "cm_pairwise_quantile_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Printf

const NFAIL = Ref(0); const NPASS = Ref(0)
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    ok ? (NPASS[] += 1) : (NFAIL[] += 1)
    println(ok ? "  PASS  " : "  FAIL  ", name, isempty(detail) ? "" : "   [$detail]")
    flush(stdout)
    return ok
end
"try/catch in a HARD scope (a top-level `catch` assignment is soft-scoped and Julia 1.12 silently
makes it a new local when a global of the same name exists)."
function threw_matching(f, fragment::AbstractString)
    try
        f(); return (false, "<did not throw>")
    catch e
        msg = sprint(showerror, e)
        return (occursin(fragment, msg), first(msg, 200))
    end
end

println("=== CM + pairwise-quantile driver ARGUMENT-TIME guards ===")
const CKPT_DIR = mktempdir(; prefix = "cmpq_argguard_")
const GRAV = default_gravity_exclude_cells_brazil_korea()
const SCI = (W = 20000, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
             σHat = 3.0, inner_lower_limit = -10.0, z_halfwidth = 2.0,
             destination_sample = :exclude_row, exclude_diagonal_gravity = true,
             gravity_exclude_cells = GRAV)
const FAMKW = (L = 5, cm_grid_size = 50, cm_moment_families = 2, contrasts = :orthonormal,
               min_bin_count = 1, mass_start = :uniform, inner_opt = "ek_inner_cmpq.opt")
# A w0 of the RIGHT length would need a real context; these guards all fire before w0 is looked at,
# so a dummy is fine and keeps the file context-free.
const W0_DUMMY = zeros(384)

run_it(; sci = SCI, fam = FAMKW, extra = NamedTuple()) =
    run_cm_pairwise_quantile_upper_checkpointed(W0_DUMMY; find_smallest = true, sci..., fam...,
        ckpt_dir = CKPT_DIR, label = "argguard", maxtime_real = 5.0, run_id = "argguard",
        verbose = false, extra...)

println("\n--- the inner option file ---")
ok, msg = threw_matching(() -> run_it(fam = merge(FAMKW, (inner_opt = "ek_inner_cmpq_fgonly.opt",))),
                         "hessopt=exact")
check("an FG-only option file is refused AT ARGUMENT TIME, before any context build", ok, msg)
# NOT a check that ek_inner.opt is refused: it DOES request `hessopt exact` (verified), so the
# guard correctly lets it through. The reason production must use ek_inner_cmpq.opt is its
# TOLERANCES and LINEAR SOLVER -- opttol=opttol_abs=1e-10 (1e-12 sits below the ~1e-11 achievable
# floor), linsolver ma97 with 4 threads (`auto` picks a serial solver), maxit 100 -- all measured on
# this family's own KKT structure. That is a performance/tolerance choice, not a correctness gate,
# and this file does not pretend otherwise.
ok, msg = threw_matching(() -> run_it(fam = merge(FAMKW, (inner_opt = "no_such_file.opt",))),
                         "does not exist")
check("a nonexistent option file is refused with a path, not a MethodError", ok, msg)

println("\n--- the restriction's structural condition ---")
for badL in (3, 4, 7)
    ok, msg = threw_matching(() -> run_it(fam = merge(FAMKW, (L = badL,))), "does not divide")
    check("L=$badL does not divide G=50 and is refused", ok, msg)
end
ok, msg = threw_matching(() -> run_it(fam = merge(FAMKW, (L = 1,))), "must be >= 2")
check("L=1 is refused (a single bin is not a restriction)", ok, msg)
# ...and the values that DO divide 50 must be ACCEPTED. A guard that rejected everything would pass
# every check above vacuously. Checked against `resolve_cm_pairwise_quantile_config` directly rather
# than through the driver: driving them through `run_it` would get past the divisibility guard and
# then spend ~80s building a real D=20 context per value, only to fail on the dummy w0 -- eighty
# seconds to learn nothing this line does not already establish.
for goodL in (2, 5, 10, 25)
    rc = resolve_cm_pairwise_quantile_config(
        CMPairwiseQuantileConfig(L = goodL, cm_grid_size = 50, cm_moment_families = 2,
                                 contrasts = :orthonormal, min_bin_count = 1, mass_start = :uniform))
    check("L=$goodL DIVIDES G=50 and resolves cleanly", rc.L == goodL && rc.G == 50,
          "n_cm_levels=$(rc.n_cm_levels)")
end

println("\n--- enumerated kwargs ---")
for (nm, kwmod, frag) in (("contrasts", (contrasts = :bogus,), "contrasts must be"),
                          ("cm_moment_families", (cm_moment_families = 3,), "cm_moment_families must be"),
                          ("mass_start", (mass_start = :bogus,), "mass_start must be"),
                          ("min_bin_count", (min_bin_count = 0,), "min_bin_count must be"))
    ok, msg = threw_matching(() -> run_it(fam = merge(FAMKW, kwmod)), frag)
    check("an invalid $nm is refused", ok, msg)
end
ok, msg = threw_matching(() -> run_it(sci = merge(SCI, (destination_sample = :bogus,))),
                         "destination_sample must be")
check("an invalid destination_sample is refused", ok, msg)
# The joint-cell arithmetic floor: L^2 cells must be satisfiable by W before any data is read.
ok, msg = threw_matching(() -> run_it(sci = merge(SCI, (W = 10,)),
                                      fam = merge(FAMKW, (min_bin_count = 100,))), "unsatisfiable")
check("an arithmetically unsatisfiable min_bin_count is refused before any data is read", ok, msg)

println("\n--- no silent defaults on any scientific kwarg ---")
for nm in keys(SCI)
    sci_missing = NamedTuple(k => v for (k, v) in pairs(SCI) if k != nm)
    ok, _ = threw_matching(() -> run_it(sci = sci_missing), "UndefKeywordError")
    check("omitting the scientific kwarg $nm raises UndefKeywordError", ok)
end
for nm in keys(FAMKW)
    fam_missing = NamedTuple(k => v for (k, v) in pairs(FAMKW) if k != nm)
    ok, _ = threw_matching(() -> run_it(fam = fam_missing), "UndefKeywordError")
    check("omitting this family's kwarg $nm raises UndefKeywordError", ok)
end
ok, _ = threw_matching(() -> run_cm_pairwise_quantile_upper_checkpointed(W0_DUMMY; SCI..., FAMKW...,
            ckpt_dir = CKPT_DIR, label = "argguard", maxtime_real = 5.0), "UndefKeywordError")
check("omitting find_smallest raises UndefKeywordError (direction is never defaulted)", ok)

rm(CKPT_DIR; recursive = true, force = true)
println("\n", "="^92)
@printf("TOTAL: %d passed, %d FAILED\n", NPASS[], NFAIL[])
println("="^92)
NFAIL[] == 0 || error("test_cm_pairwise_quantile_driver_argguards: $(NFAIL[]) check(s) failed")
